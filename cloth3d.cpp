#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstdio> // For snprintf, a function C++ programmers pretend not to use.
#include <unordered_map>

// Look at me, using a fancy font-rendering library. This is where the magic (and a whole lot of dependencies) happens.
#include <map>
#include <ft2build.h>
#include FT_FREETYPE_H

// GLM, because doing matrix math by hand is a form of self-harm I'm not ready for.
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// The one-header library to rule them all. Praise be to Sean Barrett.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// I guess these are reasonable window dimensions. Who even has a 4:3 monitor anymore? Oh, wait. This isn't 4:3. My math is great.
const int WINDOW_WIDTH  = 1280;
const int WINDOW_HEIGHT = 1080;

// Struct: SimParams
// A neat little package for all the numbers we let the user fiddle with.
// This is basically the control panel for our chaotic fabric universe.
struct SimParams {
    int   accuracy; // How many times we try to fix the cloth's mistakes each frame. More = Slower.
    float gravity;  // How aggressively we want to pull things downward. Named after a popular movie.
    int   clothY;   // How many points tall the cloth is. More points, more droop.
    int   clothX;   // How many points wide it is.
    float spacing;  // The distance between points. The "resolution" of our fabric.
    float tearDist; // How far points can stretch before the link between them snaps. For dramatic effect.
    float friction; // How much velocity the cloth loses. 1.0 is no friction, 0.9 is "moving through molasses".
    float bounce;   // How bouncy the floor is. 0.0 is a sad splat, 1.0 is a superball.
};

// Global State Management for the UI. Yes, they're global. No, I don't want to talk about it.
SimParams currentParams; // The parameters the simulation is *actually* using.
SimParams uiParams;      // The parameters the user is editing, but hasn't applied yet. A sandbox of chaos.
int selectedParam = 0;   // Which parameter in the UI is currently highlighted.
const int NUM_PARAMS = 8;  // The total number of things you can break.

/*
 = *=================================
 Shader Source Code
 ==================================
 Here be dragons and incomprehensible GLSL magic.
 I copied most of this from a tutorial. If it works, don't touch it.
 If it doesn't work, well, good luck.
 */

// Vertex Shader: The first stage of grief.
// It takes our 3D vertex positions and projects them into the 2D space of your screen.
const char* vertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)glsl";

// Fragment Shader: The final coloring book stage.
// It decides the color of each pixel, doing some fancy math for lighting.
const char* fragmentShaderSource = R"glsl(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D clothTexture;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

void main() {
    float ambientStrength = 0.8;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0.7;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 lighting = (ambient + diffuse + specular);
    vec4 texColor = texture(clothTexture, TexCoord);

    if(texColor.a < 0.1) discard; // If the texture is transparent, just yeet the pixel.

    FragColor = vec4(lighting * texColor.rrr, texColor.a);
}
)glsl";

// UI Vertex Shader: A much simpler version for drawing 2D text.
// No 3D, no lighting, just "put this character here, please".
const char* uiVertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
out vec2 TexCoords;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}
)glsl";

// UI Fragment Shader: Also simple. Takes a single-channel (red) texture from FreeType
// and colors it. The text color is a uniform, so we can change it on the fly.
const char* uiFragmentShaderSource = R"glsl(
#version 330 core
in vec2 TexCoords;
out vec4 color;
uniform sampler2D text;
uniform vec3 textColor;
void main() {
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
    color = vec4(textColor, 1.0) * sampled;
}
)glsl";


// Class: Shader
// Our personal translator for talking to the GPU.
// It wraps up all the tedious OpenGL calls for compiling and using shader programs.
class Shader {
public:
    GLuint ID; // The shader program's ID. It's just a number, but it's *our* number.

    // Function: compile
    // Takes our human-readable-ish shader code and turns it into something the graphics card might understand.
    // Pray for no compiler errors.
    void compile(const char* vSrc, const char* fSrc) {
        GLuint sV, sF;
        sV = glCreateShader(GL_VERTEX_SHADER); glShaderSource(sV, 1, &vSrc, NULL); glCompileShader(sV); checkErrors(sV, "VERTEX");
        sF = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(sF, 1, &fSrc, NULL); glCompileShader(sF); checkErrors(sF, "FRAGMENT");
        ID = glCreateProgram(); glAttachShader(ID, sV); glAttachShader(ID, sF); glLinkProgram(ID); checkErrors(ID, "PROGRAM");
        glDeleteShader(sV); glDeleteShader(sF); // We don't need the originals anymore, they live on in the program.
    }

    // Function: use
    // Tells OpenGL, "Hey, use this program I just compiled! No, the *other* one. Yes, that one. Please?"
    Shader& use() { glUseProgram(ID); return *this; }

    // Functions: setMat4, setVec3, setInt
    // Shoveling data into the shader's gaping maw. Here's a matrix, here's a vector... are you happy now, GPU?
    void setMat4(const char* n, const glm::mat4& m) { glUniformMatrix4fv(glGetUniformLocation(ID, n), 1, GL_FALSE, glm::value_ptr(m)); }
    void setVec3(const char* n, const glm::vec3& v) { glUniform3fv(glGetUniformLocation(ID, n), 1, &v[0]); }
    void setInt(const char* n, int v) { glUniform1i(glGetUniformLocation(ID, n), v); }
private:
    // Function: checkErrors
    // The function of pure existential dread. Did it work? Or did I just waste another hour of my life?
    // It checks for shader compilation and linking errors and prints them to the console.
    void checkErrors(GLuint s, std::string t) {
        GLint success; GLchar infoLog[1024];
        if (t != "PROGRAM") { glGetShaderiv(s, GL_COMPILE_STATUS, &success); if (!success) { glGetShaderInfoLog(s, 1024, NULL, infoLog); std::cout << "SHADER ERROR: " << t << "\n" << infoLog << std::endl; } }
        else { glGetProgramiv(s, GL_LINK_STATUS, &success); if (!success) { glGetProgramInfoLog(s, 1024, NULL, infoLog); std::cout << "SHADER ERROR: " << t << "\n" << infoLog << std::endl; } }
    }
};

// Function: loadTexture
// Because a plain white cloth is just sad. This loads an image from a file into an OpenGL texture.
GLuint loadTexture(const char* path) {
    GLuint texID; glGenTextures(1, &texID);
    int w, h, nc; unsigned char* data = stbi_load(path, &w, &h, &nc, 0);
    if (data) {
        GLenum format = GL_RED; if (nc == 3) format = GL_RGB; else if (nc == 4) format = GL_RGBA;
        glBindTexture(GL_TEXTURE_2D, texID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        stbi_image_free(data);
    } else {
        // If it fails, we just print an error and carry on. The cloth will look weird, but that's a problem for future me.
        std::cout << "Texture failed to load: " << path << std::endl;
    }
    return texID;
}

// Struct: Character
// A struct to hold all the glyph data from FreeType. Basically a trading card for each letter of the alphabet.
struct Character {
    GLuint      TextureID;  // The texture for this single character.
    glm::ivec2  Size;       // How big the character image is.
    glm::ivec2  Bearing;    // How to offset the character from the cursor position.
    GLuint      Advance;    // How far to move the cursor for the next character.
};


// Class: UIRenderer
// The reason you can read this text instead of binary. This class handles all the FreeType font rendering.
class UIRenderer {
public:
    GLuint textVAO, textVBO;
    Shader textShader;
    std::map<GLchar, Character> Characters; // A map to hold all our pre-rendered character data.

    // Function: init
    // Wakes up the FreeType monster, feeds it a font file (I hope 'mono.ttf' exists),
    // and forces it to spit out textures for every character. A truly herculean task that probably has a dozen failure points.
    void init() {
        textShader.compile(uiVertexShaderSource, uiFragmentShaderSource);
        glm::mat4 projection = glm::ortho(0.0f, static_cast<GLfloat>(WINDOW_WIDTH), 0.0f, static_cast<GLfloat>(WINDOW_HEIGHT));
        textShader.use();
        textShader.setInt("text", 0);
        textShader.setMat4("projection", projection);

        FT_Library ft;
        if (FT_Init_FreeType(&ft)) {
            std::cout << "ERROR::FREETYPE: Could not init FreeType Library. Well, crap." << std::endl;
            return;
        }

        // I'm assuming 'mono.ttf' is just lying around. This is a bold and probably foolish assumption.
        const char* font_path = "mono.ttf";
        FT_Face face;
        if (FT_New_Face(ft, font_path, 0, &face)) {
            std::cout << "ERROR::FREETYPE: Failed to load font from " << font_path << ". My brilliant plan is ruined." << std::endl;
            return;
        }

        FT_Set_Pixel_Sizes(face, 0, 24); // Let's make the font 24 pixels tall. Seems readable.
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // This is important for non-power-of-two textures, which fonts are full of.

        // Let's load the first 128 ASCII characters. Sorry, no emoji support today.
        for (unsigned char c = 0; c < 128; c++) {
            if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
                std::cout << "ERROR::FREETYTPE: Failed to load Glyph for char: " << c << std::endl;
                continue;
            }
            GLuint texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RED,
                face->glyph->bitmap.width, face->glyph->bitmap.rows,
                0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer
            );
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            // Store this character's data in our map for later.
            Character character = {
                texture,
                glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                static_cast<GLuint>(face->glyph->advance.x)
            };
            Characters.insert(std::pair<GLchar, Character>(c, character));
        }
        glBindTexture(GL_TEXTURE_2D, 0);

        // We're done with FreeType for now. Clean up its mess.
        FT_Done_Face(face);
        FT_Done_FreeType(ft);

        // Configure the VAO/VBO for drawing the character quads. We'll reuse this for every character.
        glGenVertexArrays(1, &textVAO);
        glGenBuffers(1, &textVBO);
        glBindVertexArray(textVAO);
        glBindBuffer(GL_ARRAY_BUFFER, textVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    // Function: drawText
    // Takes a string and draws it on screen, one character at a time.
    // It's like a tiny, inefficient digital printing press. The kerning is probably terrible, but who's looking that closely?
    void drawText(std::string text, GLfloat x, GLfloat y, GLfloat scale, glm::vec3 color) {
        textShader.use();
        textShader.setVec3("textColor", color);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(textVAO);

        // Iterate through each character in the string...
        for (auto c = text.begin(); c != text.end(); c++) {
            Character ch = Characters[*c];

            GLfloat xpos = x + ch.Bearing.x * scale;
            GLfloat ypos = y - (ch.Size.y - ch.Bearing.y) * scale;
            GLfloat w = ch.Size.x * scale;
            GLfloat h = ch.Size.y * scale;

            // Update the VBO with the geometry for this specific character.
            GLfloat vertices[6][4] = {
                { xpos,     ypos + h,   0.0, 0.0 }, { xpos,     ypos,       0.0, 1.0 }, { xpos + w, ypos,       1.0, 1.0 },
                { xpos,     ypos + h,   0.0, 0.0 }, { xpos + w, ypos,       1.0, 1.0 }, { xpos + w, ypos + h,   1.0, 0.0 }
            };

            // Slap the character's texture onto the quad.
            glBindTexture(GL_TEXTURE_2D, ch.TextureID);
            glBindBuffer(GL_ARRAY_BUFFER, textVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDrawArrays(GL_TRIANGLES, 0, 6); // Draw the quad!

            // Advance the cursor for the next character. The advance is in 1/64ths of a pixel, so we bitshift. Magic!
            x += (ch.Advance >> 6) * scale;
        }
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
};


// Struct: Vec2
// Just two floats. Used for 2D texture coordinates. Not much to see here. Move along.
struct Vec2 { float x,y; Vec2(float x=0,float y=0):x(x),y(y){} };

// Struct: Vec3
// Three floats! Now we're in the big leagues. Comes with all the math operators you'd expect.
// I even remembered to add the `operator*=` this time. Character development.
struct Vec3 {
    float x,y,z; Vec3(float x=0,float y=0,float z=0):x(x),y(y),z(z){}
    Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x,y+o.y,z+o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x,y-o.y,z-o.z); }
    Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); }
    Vec3& operator+=(const Vec3& o) { x+=o.x;y+=o.y;z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x-=o.x;y-=o.y;z-=o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; } // This was missing before. We don't talk about that.
    float length() const { return std::sqrt(x*x+y*y+z*z); }
    Vec3 cross(const Vec3& o) const { return Vec3(y*o.z-z*o.y, z*o.x-x*o.z, x*o.y-y*o.x); }
    Vec3 normalized() const { float l=length(); return(l>0)?Vec3(x/l,y/l,z/l):Vec3(0,0,0); }
};

// Struct: Vertex
// A bundle of data we throw at the GPU. It has a position, a normal (for fancy lighting), and texture coordinates.
// It's the atom of our 3D world.
struct Vertex { Vec3 Position; Vec3 Normal; Vec2 TexCoords; };

// Global variables for camera and mouse.
// Because who needs proper encapsulation when you can just make everything global? It's fine. This is fine.
float cameraRotX=-25.0f,cameraRotY=30.0f,zoom=1.0f;
glm::vec3 cameraPos; bool cameraRotating=false; double lastMouseX, lastMouseY;
struct Mouse { float cut=8.0f,influence=36.0f; bool down=false; int button=0; float x=0,y=0; } mouse;

// Forward declarations because C++ enjoys making us type things twice.
class Point;
class Constraint;

// Class: Constraint
// The digital rubber bands holding our cloth together. Each constraint connects two points.
class Constraint {
public:
    // We store indices into the main points vector instead of raw pointers.
    // Why? Because pointers are scary and would probably invalidate themselves, causing a spectacular crash. This is safer. Maybe.
    size_t p1_idx, p2_idx;
    float restLength; // The length this constraint *wants* to be.
    bool to_be_removed = false; // A flag for when the user cuts the cloth or it tears.

    Constraint(size_t idx1, size_t idx2, std::vector<Point>& allPoints);
    void resolve(std::vector<Point>& allPoints);
};

// Class: Point
// A single, lonely particle in the physics simulation. It has a position, a previous position, and not much else.
class Point {
public:
    Vec3 pos, prevPos, normal; // The secret to Verlet integration is storing where the point is AND where it was.
    Vec2 texCoord;
    bool pinned = false; // Is this point nailed in place?
    Vec3 pinPos;

    Point(float x, float y, float z) : pos(x, y, z), prevPos(x, y, z) {}
    void update(float dt, const glm::mat4& v, const glm::mat4& p, int& cut_point_idx, size_t my_idx, const std::vector<int>& pointConstraintCount);
};

// Constructor: Constraint
// Calculates the initial resting length between two points when the cloth is first created.
Constraint::Constraint(size_t idx1, size_t idx2, std::vector<Point>& allPoints) : p1_idx(idx1), p2_idx(idx2) {
    restLength = (allPoints[idx1].pos - allPoints[idx2].pos).length();
}

// Function: Constraint::resolve
// The heart of the simulation. It looks at two points, checks if they're too far apart or too close,
// and then nudges them. Repeat this a few thousand times per frame and you've got something that looks like cloth!
void Constraint::resolve(std::vector<Point>& allPoints) {
    Point& p1 = allPoints[p1_idx];
    Point& p2 = allPoints[p2_idx];

    Vec3 diff = p1.pos - p2.pos;
    float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

    if (distSq < 1e-6f) return; // If they're basically on top of each other, don't divide by zero.

    float dist = std::sqrt(distSq);

    // If the distance is greater than our "tear" threshold, flag this constraint for deletion. R.I.P.
    if (dist > currentParams.tearDist) {
        to_be_removed = true;
        return;
    }

    float difference = (restLength - dist) / dist;
    Vec3 correction = diff * (difference * 0.5f);

    // Apply the correction, but only if the point isn't pinned down.
    if (!p1.pinned) p1.pos += correction;
    if (!p2.pinned) p2.pos -= correction;
}

// Function: Point::update
// Does all the physics-y stuff for a single point each frame.
void Point::update(float dt, const glm::mat4& v, const glm::mat4& p, int& cut_point_idx, size_t my_idx, const std::vector<int>& pointConstraintCount) {
    if (pinned) {
        pos = pinPos; // If pinned, don't move. At all.
        return;
    }

    // Check for mouse interaction. Are we pulling or cutting?
    if (mouse.down) {
        glm::vec4 vp(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        // Project the 3D point to 2D screen space to see if the mouse is near it.
        glm::vec3 sp = glm::project(glm::vec3(pos.x, pos.y, pos.z), v, p, vp);
        float smY = float(WINDOW_HEIGHT) - float(mouse.y); // Screen coordinates are weird.
        float dx = sp.x - mouse.x, dy = sp.y - smY;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (mouse.button == 1 && dist < mouse.influence) {
            // Left-click drag: Un-project the mouse's 2D position back into 3D space and pull the point there.
            glm::vec3 wp = glm::unProject(glm::vec3(mouse.x, smY, sp.z), v, p, vp);
            pos = Vec3(wp.x, wp.y, wp.z);
        } else if (mouse.button == 2 && dist < mouse.cut) {
            // Right-click: This point is marked for death (or at least, its constraints are).
            cut_point_idx = static_cast<int>(my_idx);
        }
    }

    // Verlet integration step. So simple, it feels like it shouldn't work.
    Vec3 accel = Vec3(0, currentParams.gravity, 0);
    Vec3 velocity = pos - prevPos;
    Vec3 temp = pos;
    pos = pos + velocity * currentParams.friction + accel * dt * dt;
    prevPos = temp;

    // A simple, invisible floor to stop the cloth from falling forever.
    if (pos.y < -200.0f) {
        pos.y = -200.0f;
        // The bounce factor makes it lose some energy on impact.
        prevPos.y = pos.y + (pos.y - prevPos.y) * currentParams.bounce;
    }
}

// Function: compute_triangle_normal
// Does some vector cross-product magic to figure out which way a triangle is facing.
// Essential for making the lighting look not-terrible.
Vec3 compute_triangle_normal(const Point& p1, const Point& p2, const Point& p3) {
    return (p2.pos - p1.pos).cross(p3.pos - p1.pos).normalized();
}

// Function: hashGrid
// A fancy function to turn a 3D coordinate into a single number for our spatial hash grid.
// It uses bit-shifting and magic numbers I definitely copied from the internet.
// I have no idea how it works, but it's fast, and that's all that matters.
inline size_t hashGrid(int x, int y, int z) {
    size_t hash = 0;
    hash ^= static_cast<size_t>(x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= static_cast<size_t>(y) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= static_cast<size_t>(z) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
}

// Struct: CollisionConstants
// A home for all the pre-calculated values for collision detection.
// Why? So we don't have to recalculate them every frame, duh. This is what peak optimization looks like.
struct CollisionConstants {
    float cellSize;
    float spacingSquared;
    float invCellSize;
    int frameSkip;

    void update(float spacing) {
        cellSize = spacing * 2.0f;
        spacingSquared = spacing * spacing;
        invCellSize = 1.0f / cellSize;
        frameSkip = 1;
    }
};

// Class: Cloth
// The main event. The whole enchilada. The fabric of our digital lives.
class Cloth {
public:
    std::vector<Point> points;
    std::vector<Constraint> constraints;
    std::vector<Vertex> renderVertices;
    std::vector<unsigned int> indices;
    GLuint vao, vbo, ebo;
    int cut_point_idx = -1; // Which point is being cut, if any. -1 means none.

    CollisionConstants collisionConsts;

private:
    // A map that acts as a 3D grid to speed up self-collision checks.
    std::map<long long, std::vector<size_t>> grid;
    void resolveSelfCollisions();

public:
    // Constructor: Cloth
    // Lays out all the points in a nice grid and then connects them with constraints,
    // like a digital knitting circle. Also pins the top row so it doesn't just fall into the abyss.
    Cloth(const SimParams& params) {
        float sX = -params.clothX * params.spacing * 0.5f, sY = 300.0f, sZ = 0.0f;
        for (int y = 0; y <= params.clothY; ++y) {
            for (int x = 0; x <= params.clothX; ++x) {
                // We create points directly in the vector. It's supposedly faster.
                points.emplace_back(sX + x * params.spacing, sY - y * params.spacing, sZ);
                Point& p = points.back();
                p.texCoord = Vec2((float)x / params.clothX, (float)y / params.clothY);

                // Now create constraints using INDICES. Much safer than pointers.
                size_t currentIndex = points.size() - 1;
                if (x > 0) { // Connect to the point to the left.
                    constraints.emplace_back(currentIndex, currentIndex - 1, points);
                }
                if (y > 0) { // Connect to the point above.
                    constraints.emplace_back(currentIndex, currentIndex - (params.clothX + 1), points);
                }
                // Pin the top row of points in place.
                if (y == 0) { p.pinned = true; p.pinPos = p.pos; }
            }
        }
        collisionConsts.update(params.spacing);
        initDrawingData(params);
    }

    // Destructor: Cloth
    // The destroyer. Cleans up the OpenGL buffers we so carelessly allocated.
    ~Cloth() {
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ebo);
    }

    // Function: initDrawingData
    // Sets up the VAO, VBO, and EBO. This is a bunch of OpenGL boilerplate that
    // you write once and pray you never have to debug. It defines how the vertices are structured for rendering.
    void initDrawingData(const SimParams& params) {
        // Create the triangles from the grid of points.
        for (int y = 0; y < params.clothY; ++y) for (int x = 0; x < params.clothX; ++x) {
            int p1 = y * (params.clothX + 1) + x, p2 = p1 + 1, p3 = p1 + params.clothX + 1, p4 = p3 + 1;
            indices.push_back(p1); indices.push_back(p2); indices.push_back(p3);
            indices.push_back(p3); indices.push_back(p2); indices.push_back(p4);
        }
        renderVertices.resize(points.size());
        glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo); glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, renderVertices.size() * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Position));
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
        glBindVertexArray(0);
    }

    // Function: computeNormals
    // Calculates the normal for each vertex so the lighting looks smooth and not like a jagged mess.
    // It averages the normals of all the triangles connected to a point. It's a hack, but it works.
    void computeNormals() {
        for (auto& p : points) p.normal = Vec3(0, 0, 0);
        for (size_t i = 0; i < indices.size(); i += 3) {
            Point& p1 = points[indices[i]];
            Point& p2 = points[indices[i+1]];
            Point& p3 = points[indices[i+2]];
            Vec3 n = compute_triangle_normal(p1, p2, p3);
            p1.normal += n; p2.normal += n; p3.normal += n;
        }
        for (auto& p : points) p.normal = p.normal.normalized();
    }

    // Function: updateGpuBuffers
    // Shovels the latest point positions and normals over to the GPU.
    // This has to happen every frame, or we'd just be staring at a static piece of cloth.
    void updateGpuBuffers() {
        for (size_t i = 0; i < points.size(); ++i) {
            renderVertices[i].Position = points[i].pos;
            renderVertices[i].Normal = points[i].normal;
            renderVertices[i].TexCoords = points[i].texCoord;
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, renderVertices.size() * sizeof(Vertex), renderVertices.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Function: update
    // The main brain of the cloth simulation. It calls the constraint solver, handles collisions,
    // updates all the points, and checks if you're trying to cut the cloth.
    void update(float dt, const glm::mat4& v, const glm::mat4& p) {
        // We might do fewer constraint iterations if the cloth isn't moving much. A cheap optimization.
        int actualAccuracy = currentParams.accuracy;
        static int stableFrames = 0;
        static float lastTotalMovement = 0.0f;
        float totalMovement = 0.0f;
        for (const auto& point : points) {
            if (!point.pinned) totalMovement += (point.pos - point.prevPos).length();
        }
        if (totalMovement < lastTotalMovement * 1.1f) {
            if (++stableFrames > 10) actualAccuracy = std::max(3, actualAccuracy / 2);
        } else {
            stableFrames = 0;
        }
        lastTotalMovement = totalMovement;

        // Resolve structural constraints over and over again.
        for (int i = 0; i < actualAccuracy; ++i) {
            for (auto& c : constraints) c.resolve(points);
        }

        // Resolve self-collisions to prevent clipping.
        resolveSelfCollisions();

        // Update each point's position based on physics.
        cut_point_idx = -1;
        static std::vector<int> pointConstraintCount; // static to avoid reallocating memory every frame.
        pointConstraintCount.assign(points.size(), 0);
        for (const auto& c : constraints) {
            if (!c.to_be_removed) {
                pointConstraintCount[c.p1_idx]++;
                pointConstraintCount[c.p2_idx]++;
            }
        }
        for (size_t i = 0; i < points.size(); ++i) {
            points[i].update(dt, v, p, cut_point_idx, i, pointConstraintCount);
        }

        // If a point was marked to be cut, flag all its constraints for removal.
        if (cut_point_idx != -1) {
            for (auto& c : constraints) {
                if (c.p1_idx == cut_point_idx || c.p2_idx == cut_point_idx) c.to_be_removed = true;
            }
        }

        // The "erase-remove" idiom. A fancy way to delete all the flagged constraints from the vector.
        auto newEnd = std::remove_if(constraints.begin(), constraints.end(), [](const Constraint& c) { return c.to_be_removed; });
        if (newEnd != constraints.end()) constraints.erase(newEnd, constraints.end());
    }

    // Function: draw
    // The artist. Tells OpenGL exactly how to draw the cloth, which shaders to use, where the lights are, etc.
    // It also draws the wireframe, because that looks cool and makes me feel like a real programmer.
    void draw(Shader& shader, GLuint texID, const glm::mat4& view, const glm::mat4& projection) {
        computeNormals();
        updateGpuBuffers();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0); // Pushes the filled polygons back a bit so the wireframe doesn't z-fight.

        // Rebuild the index buffer on the fly to hide triangles that have been torn apart.
        std::vector<unsigned int> dynamic_indices;
        dynamic_indices.reserve(indices.size());
        for (size_t i = 0; i < indices.size(); i += 3) {
            Point& p1 = points[indices[i]], &p2 = points[indices[i+1]], &p3 = points[indices[i+2]];
            if ((p1.pos-p2.pos).length()<currentParams.tearDist && (p2.pos-p3.pos).length()<currentParams.tearDist && (p3.pos-p1.pos).length()<currentParams.tearDist) {
                dynamic_indices.push_back(indices[i]);
                dynamic_indices.push_back(indices[i + 1]);
                dynamic_indices.push_back(indices[i + 2]);
            }
        }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, dynamic_indices.size() * sizeof(unsigned int), dynamic_indices.data(), GL_DYNAMIC_DRAW);

        // Set up the shader with all our uniforms.
        shader.use();
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        shader.setMat4("model", glm::mat4(1.0f));
        shader.setVec3("lightPos", glm::vec3(200.0f, 500.0f, 200.0f));
        shader.setVec3("viewPos", cameraPos);
        shader.setVec3("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
        shader.setInt("clothTexture", 0);

        // Bind the texture and draw the cloth!
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texID);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, dynamic_indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glDisable(GL_POLYGON_OFFSET_FILL);

        // Now, for the cool part: draw the wireframe using ancient, deprecated OpenGL.
        // I'm not setting up a whole separate shader for this. Don't judge me.
        glUseProgram(0);
        glMatrixMode(GL_PROJECTION); glLoadMatrixf(glm::value_ptr(projection));
        glMatrixMode(GL_MODELVIEW); glLoadMatrixf(glm::value_ptr(view));
        glColor3f(0.2f, 0.2f, 0.2f);
        glBegin(GL_LINES);
        for (const auto& c : constraints) {
            if (!c.to_be_removed) {
                const Point& p1 = points[c.p1_idx];
                const Point& p2 = points[c.p2_idx];
                glVertex3f(p1.pos.x, p1.pos.y, p1.pos.z);
                glVertex3f(p2.pos.x, p2.pos.y, p2.pos.z);
            }
        }
        glEnd();
    }
};

int selfCollisionFrameCounter = 0;

// Function: Cloth::resolveSelfCollisions
// The most cursed function in this file. Tries to stop the cloth from passing through itself
// using a spatial hash grid. It's a mess of optimizations, magic numbers, and desperate hope.
// It mostly works. Don't look at it too hard or it might break.
void Cloth::resolveSelfCollisions() {
    // Only update our collision constants if the user actually changed the spacing.
    static float lastSpacing = -1.0f;
    if (lastSpacing != currentParams.spacing) {
        collisionConsts.update(currentParams.spacing);
        lastSpacing = currentParams.spacing;
    }

    // Skip this whole process on some frames to save performance. It's probably fine.
    selfCollisionFrameCounter = (selfCollisionFrameCounter + 1) % collisionConsts.frameSkip;
    if (selfCollisionFrameCounter != 0) return;

    // A static grid to avoid reallocating a giant map every single frame.
    static std::unordered_map<size_t, std::vector<size_t>> grid;
    for (auto& pair : grid) pair.second.clear(); // Clear the vectors, but keep the memory.
    if (collisionConsts.cellSize <= 0.0f) return;
    grid.reserve(points.size() / 4); // A wild guess at how many cells we'll need.

    // Put every point into the grid based on its position.
    for (size_t i = 0; i < points.size(); ++i) {
        const Point& p = points[i];
        if (p.pinned) continue; // Pinned points don't move, so they can't collide.
        int ix = static_cast<int>(p.pos.x * collisionConsts.invCellSize);
        int iy = static_cast<int>(p.pos.y * collisionConsts.invCellSize);
        int iz = static_cast<int>(p.pos.z * collisionConsts.invCellSize);
        grid[hashGrid(ix, iy, iz)].push_back(i);
    }

    // Now, for each point, check its own grid cell and its neighbors for other points that are too close.
    for (size_t p1_idx = 0; p1_idx < points.size(); ++p1_idx) {
        Point& p1 = points[p1_idx];
        if (p1.pinned) continue;

        int ix = static_cast<int>(p1.pos.x * collisionConsts.invCellSize);
        int iy = static_cast<int>(p1.pos.y * collisionConsts.invCellSize);
        int iz = static_cast<int>(p1.pos.z * collisionConsts.invCellSize);

        // A pre-defined list of neighbor cells to check. We only need to check forward to avoid double-checking pairs.
        static const int offsets[][3] = { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,0}, {1,0,1}, {0,1,1}, {1,1,1} };

        for (const auto& offset : offsets) {
            auto it = grid.find(hashGrid(ix + offset[0], iy + offset[1], iz + offset[2]));
            if (it == grid.end()) continue;

            for (size_t p2_idx : it->second) {
                if (p1_idx >= p2_idx) continue; // Avoid checking a point against itself or checking pairs twice.
                Point& p2 = points[p2_idx];
                if (p2.pinned) continue;

                // Check the distance and apply a correction if they're colliding.
                Vec3 diff = p1.pos - p2.pos;
                float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                if (distSq > 0.0f && distSq < collisionConsts.spacingSquared) {
                    float invDist = 1.0f / std::sqrt(distSq); // Fast-ish inverse square root.
                    float difference = (currentParams.spacing * invDist - 1.0f) * 0.5f;
                    Vec3 correction = diff * difference;
                    p1.pos += correction;
                    p2.pos -= correction;
                }
            }
        }
    }
}


// Global pointers and objects. Yes, more globals. It's a small project, forgive me.
GLFWwindow* g_window;
char g_clothMem[sizeof(Cloth)]; // Pre-allocate memory on the stack for the cloth object.
Cloth* g_cloth;                 // A pointer to our cloth, placed in the memory above.
Shader g_shader;
GLuint g_texture;
UIRenderer g_uiRenderer;


// Function: drawUI
// Draws all the text on the screen. The FPS counter, the parameters you can tweak,
// and the controls. It's a miracle this is aligned at all.
void drawUI() {
    // FPS Counter: My favorite part. Lets me know how badly my code is performing.
    static double lastTime = glfwGetTime();
    static int nbFrames = 0;
    static char fpsText[32] = "FPS: 0";
    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - lastTime >= 1.0) {
        snprintf(fpsText, sizeof(fpsText), "FPS: %d", nbFrames);
        nbFrames = 0;
        lastTime += 1.0;
    }
    g_uiRenderer.drawText(fpsText, WINDOW_WIDTH - 150, WINDOW_HEIGHT - 30, 1.0f, glm::vec3(1.0f, 1.0f, 0.0f));

    // Draw the list of tunable parameters. The selected one is yellow.
    char buffer[128];
    const int start_y_params = WINDOW_HEIGHT - 60;
    const int line_h = 25;
    snprintf(buffer, sizeof(buffer), "Accuracy: %d", uiParams.accuracy);
    g_uiRenderer.drawText(buffer, 10, start_y_params, 1.0f, selectedParam == 0 ? glm::vec3(1,1,0) : glm::vec3(1,1,1));
    snprintf(buffer, sizeof(buffer), "Gravity: %.1f", uiParams.gravity);
    g_uiRenderer.drawText(buffer, 10, start_y_params - line_h, 1.0f, selectedParam == 1 ? glm::vec3(1,1,0) : glm::vec3(1,1,1));
    snprintf(buffer, sizeof(buffer), "Cloth X: %d", uiParams.clothX);
    g_uiRenderer.drawText(buffer, 10, start_y_params - line_h * 2, 1.0f, selectedParam == 2 ? glm::vec3(1,1,0) : glm::vec3(1,1,1));
    snprintf(buffer, sizeof(buffer), "Cloth Y: %d", uiParams.clothY);
    g_uiRenderer.drawText(buffer, 10, start_y_params - line_h * 3, 1.0f, selectedParam == 3 ? glm::vec3(1,1,0) : glm::vec3(1,1,1));
    snprintf(buffer, sizeof(buffer), "Spacing: %.1f", uiParams.spacing);
    g_uiRenderer.drawText(buffer, 10, start_y_params - line_h * 4, 1.0f, selectedParam == 4 ? glm::vec3(1,1,0) : glm::vec3(1,1,1));
    snprintf(buffer, sizeof(buffer), "Tear Dist: %.1f", uiParams.tearDist);
    g_uiRenderer.drawText(buffer, 10, start_y_params - line_h * 5, 1.0f, selectedParam == 5 ? glm::vec3(1,1,0) : glm::vec3(1,1,1));
    snprintf(buffer, sizeof(buffer), "Friction: %.3f", uiParams.friction);
    g_uiRenderer.drawText(buffer, 10, start_y_params - line_h * 6, 1.0f, selectedParam == 6 ? glm::vec3(1,1,0) : glm::vec3(1,1,1));
    snprintf(buffer, sizeof(buffer), "Bounce: %.2f", uiParams.bounce);
    g_uiRenderer.drawText(buffer, 10, start_y_params - line_h * 7, 1.0f, selectedParam == 7 ? glm::vec3(1,1,0) : glm::vec3(1,1,1));
    g_uiRenderer.drawText("Press 'R' to apply changes", 10, start_y_params - line_h * 9, 0.8f, glm::vec3(0.7f));


    // Draw the list of controls on the other side of the screen.
    const float controls_x = WINDOW_WIDTH - 380;
    const float controls_start_y = WINDOW_HEIGHT - 30 - line_h;
    g_uiRenderer.drawText("Controls:", controls_x, controls_start_y, 1.0f, glm::vec3(1.0f, 1.0f, 0.7f));
    g_uiRenderer.drawText("L-drag: pull", controls_x, controls_start_y - line_h, 1.0f, glm::vec3(1.0f));
    g_uiRenderer.drawText("R-click: cut", controls_x, controls_start_y - line_h * 2, 1.0f, glm::vec3(1.0f));
    g_uiRenderer.drawText("Shift-drag: rotate", controls_x, controls_start_y - line_h * 3, 1.0f, glm::vec3(1.0f));
    g_uiRenderer.drawText("Scroll: zoom", controls_x, controls_start_y - line_h * 4, 1.0f, glm::vec3(1.0f));
    g_uiRenderer.drawText("Arrows:", controls_x, controls_start_y - line_h * 5, 1.0f, glm::vec3(1.0f));
    g_uiRenderer.drawText("UP/Down UI Nav", controls_x, controls_start_y - line_h * 6, 1.0f, glm::vec3(1.0f));
    g_uiRenderer.drawText("Left/Right Change Value", controls_x, controls_start_y - line_h * 7, 1.0f, glm::vec3(1.0f));
}

// Function: keyCallback
// Handles all keyboard input. Moving the UI selection, changing values, and the all-important 'R'
// to reset the simulation with your new, probably-gonna-explode parameters.
void keyCallback(GLFWwindow*w, int k, int, int a, int) {
    if (a == GLFW_PRESS || a == GLFW_REPEAT) {
        switch (k) {
            case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, GLFW_TRUE); break;
            case GLFW_KEY_R: // The big red button.
                currentParams = uiParams; // Apply the new settings.
                g_cloth->~Cloth(); // Explicitly call the destructor. This feels wrong, but it's for placement new.
                new(g_cloth) Cloth(currentParams); // Construct a new cloth in the same memory location.
                break;
            case GLFW_KEY_UP: selectedParam = (selectedParam - 1 + NUM_PARAMS) % NUM_PARAMS; break;
            case GLFW_KEY_DOWN: selectedParam = (selectedParam + 1) % NUM_PARAMS; break;
            case GLFW_KEY_LEFT: case GLFW_KEY_RIGHT: // Adjust the selected parameter.
                float sign = (k == GLFW_KEY_LEFT) ? -1.0f : 1.0f;
                switch (selectedParam) {
                    case 0: uiParams.accuracy = std::max(1, uiParams.accuracy + (int)sign); break;
                    case 1: uiParams.gravity += 10.0f * sign; break;
                    case 2: uiParams.clothX = std::max(2, uiParams.clothX + (int)sign); break;
                    case 3: uiParams.clothY = std::max(2, uiParams.clothY + (int)sign); break;
                    case 4: uiParams.spacing = std::max(1.0f, uiParams.spacing + 0.5f * sign); break;
                    case 5: uiParams.tearDist = std::max(10.0f, uiParams.tearDist + 1.0f * sign); break;
                    case 6: uiParams.friction = std::max(0.0f, std::min(1.0f, uiParams.friction + 0.005f * sign)); break;
                    case 7: uiParams.bounce = std::max(0.0f, std::min(1.0f, uiParams.bounce + 0.05f * sign)); break;
                }
                break;
        }
    }
}

// Function: mouseCallback, cursorCallback, scrollCallback
// A trio of functions to handle all the mouse shenanigans. Clicking, dragging, rotating, zooming. The usual.
void mouseCallback(GLFWwindow*w,int b,int a,int m) {
    if(a==GLFW_PRESS) {
        if(m&GLFW_MOD_SHIFT) {cameraRotating=true;glfwGetCursorPos(w,&lastMouseX,&lastMouseY);}
        else {mouse.down=true;mouse.button=(b==GLFW_MOUSE_BUTTON_LEFT)?1:2;}
    } else if(a==GLFW_RELEASE) {cameraRotating=false;mouse.down=false;}
}
void cursorCallback(GLFWwindow*,double x,double y) {
    if(cameraRotating) {cameraRotY+=(x-lastMouseX)*0.5f;cameraRotX+=(y-lastMouseY)*0.5f;lastMouseX=x;lastMouseY=y;}
    mouse.x=x;mouse.y=y;
}
void scrollCallback(GLFWwindow*,double,double yoffset) { zoom*=(yoffset>0)?0.9f:1.1f; zoom=std::max(0.2f,std::min(5.0f,zoom)); }


// Function: main
// The grand conductor of this whole chaotic orchestra. It sets up GLFW, GLEW, our shaders, our UI, and the cloth itself.
// Then it enters the main loop, a frantic, never-ending cycle of updating, drawing, and polling for events
// until you finally put it out of its misery by closing the window.
int main() {
    // These are my new, improved, more physically plausible default parameters. Probably.
    currentParams = { 15, -980.0f, 50, 50, 8.0f, 70.0f, 0.985f, 0.5f };
    uiParams = currentParams;

    // Standard OpenGL/GLFW initialization boilerplate.
    glfwInit(); glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    g_window=glfwCreateWindow(WINDOW_WIDTH,WINDOW_HEIGHT,"Cloth Simulator with UI",nullptr,nullptr);
    glfwMakeContextCurrent(g_window);glfwSwapInterval(0); // VSync is for the weak. We want all the FPS.
    glfwSetKeyCallback(g_window,keyCallback);glfwSetMouseButtonCallback(g_window,mouseCallback);
    glfwSetCursorPosCallback(g_window,cursorCallback);glfwSetScrollCallback(g_window,scrollCallback);
    glewInit();

    glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.1f,0.1f,0.1f,1.0f);
    std::cout<<"Controls:\nL-drag:pull\nR-click:cut\nShift-drag:rotate\nScroll:zoom\nArrows:UI Nav\nR:reset/apply\nESC:quit\n";

    // Initialize all our major components.
    g_shader.compile(vertexShaderSource, fragmentShaderSource);
    g_uiRenderer.init();
    g_texture=loadTexture("fabric.png");
    g_cloth=new(g_clothMem)Cloth(currentParams); // Placement new! It's fancy.

    // The Main Loop. This is where we live now.
    auto lastTime=std::chrono::high_resolution_clock::now();
    while(!glfwWindowShouldClose(g_window)) {
        // Calculate delta time, but cap it so the simulation doesn't explode if we lag.
        auto now=std::chrono::high_resolution_clock::now(); float dt=std::chrono::duration<float>(now-lastTime).count();
        lastTime=now; dt=std::min(dt,0.018f);
        glfwPollEvents();

        // Set up the view and projection matrices for our 3D scene.
        glm::mat4 proj=glm::perspective(glm::radians(45.0f),(float)WINDOW_WIDTH/(float)WINDOW_HEIGHT,1.0f,5000.0f);
        glm::mat4 view=glm::translate(glm::mat4(1.0f),glm::vec3(0.0f,0.0f,-800.0f*zoom));
        view=glm::rotate(view,glm::radians(cameraRotX),glm::vec3(1.0f,0.0f,0.0f));
        view=glm::rotate(view,glm::radians(cameraRotY),glm::vec3(0.0f,1.0f,0.0f));
        cameraPos=glm::vec3(glm::inverse(view)[3]); // Find out where the camera is for lighting calculations.

        // Update and draw everything.
        g_cloth->update(dt,view,proj);

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        g_cloth->draw(g_shader,g_texture,view,proj);

        // Draw the UI on top of everything else.
        glDisable(GL_DEPTH_TEST);
        drawUI();

        glfwSwapBuffers(g_window);
    }
    // Cleanup time.
    g_cloth->~Cloth(); glfwDestroyWindow(g_window); glfwTerminate(); return 0;
}
