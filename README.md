# Tearable Cloth Simulator: Now with 100% More C++

This is my desperate, probably ill-advised attempt to recreate the wonderfully simple and elegant [Tearable Cloth simulation by dissimulate](https://github.com/dissimulate/Tearable-Cloth) in C++ and modern OpenGL.

I saw the original JavaScript implementation and thought, "I can do that, but with more pointers, manual memory management, and a deep sense of existential dread every time I have to link a new library." And so, this was born. It's a real-time, interactive cloth simulation where you can pull, tear, and generally abuse a piece of digital fabric to your heart's content.

## Features (or, "Things That Mostly Work")

*   **Verlet Integration Physics:** A surprisingly simple and stable physics model that simulates the cloth's points and constraints. It's the "it just works" of simulation physics.
*   **Interactive Chaos:**
    *   **Pull:** Grab the cloth with your mouse and drag it around.
    *   **Tear:** Right-click to cut constraints and watch it fall apart. Cathartic, isn't it?
*   **Real-time UI:** An in-window UI built with FreeType that lets you fiddle with the very fabric of reality (or at least the simulation parameters). Change gravity, cloth size, stiffness, and more without recompiling!
*   **Self-Collision (Kind Of):** A spatial hashing grid desperately tries to prevent the cloth from passing through itself. It's a valiant effort that works most of the time. Don't look at it too closely.
*   **Basic Rendering:** It's got a texture, it's got Blinn-Phong lighting, and it's even got a wireframe overlay (drawn with ancient, deprecated OpenGL, because who has time for a whole separate shader?).

## How to Get It Running (Good Luck)

This isn't a simple "download and run" situation. You're a C++ developer; you knew what you were signing up for.

### Step 0: The Missing Pieces

This repository is lonely. It's missing a couple of friends that it needs to work. Before you even *think* about compiling, you need to find:

1.  `fabric.png`: A texture for the cloth. Any square, tileable fabric image will do. You can find one on any free texture site. Just name it `fabric.png` and place it in the root directory (next to the source file).
2.  `mono.ttf`: A monospaced font file for the UI. The code is hardcoded to look for this exact file. Find any monospaced `.ttf` font on your system (like Consolas, Fira Code, DejaVu Sans Mono, etc.), rename it to `mono.ttf`, and place it in the root directory.

### Step 1: Dependencies

You'll need to have the following libraries installed and ready to be found by your compiler/linker. How you do this is your own personal journey. May the odds be ever in your favor.

*   **GLEW** (The OpenGL Extension Wrangler Library)
*   **GLFW** (A library for creating windows, contexts, and handling input)
*   **GLM** (OpenGL Mathematics)
*   **FreeType** (For the fancy text rendering)
*   **A C++ compiler** that doesn't hate you (GCC, Clang, MSVC)
*   **CMake** (To make building slightly less painful)

*(Note: `stb_image.h` is included directly in the source, so you don't need to link it separately. Praise be to the stb single-header libraries.)*

### Step 2: Compiling with CMake

Assuming you've wrestled the dependencies into submission, the build process should be standard CMake procedure.

```
# 1. Create a build directory (don't pollute the source!)
mkdir build
cd build

# 2. Run CMake to configure the project
cmake ..

# 3. Compile the code
# On Linux/macOS
make
# On Windows with Visual Studio, you might run this instead
cmake --build .

```

### Step 3: Run It!

If the stars aligned and the compiler was in a good mood, you should have an executable in your `build` directory.

**IMPORTANT:** Run the executable from the project's **root directory**, not from inside the `build` directory. It needs to be able to find `fabric.png` and `mono.ttf`.

```
# From the project root directory
./build/cloth3d
```

## Controls

Once it's running, here's how you can play with it.

| Action                  | Keys / Mouse                 |
| ----------------------- | ---------------------------- |
| **Pull Cloth**          | Left-Click and Drag          |
| **Cut Cloth**           | Right-Click                  |
| **Rotate Camera**       | `Shift` + Left-Click and Drag|
| **Zoom Camera**         | Mouse Scroll Wheel           |
| **Navigate UI**         | `Up`/`Down` Arrow Keys       |
| **Change UI Value**     | `Left`/`Right` Arrow Keys    |
| **Apply UI Changes**    | `R` Key (resets simulation)  |
| **Quit**                | `ESC` Key                    |

## A Note on Code Philosophy (or lack thereof)

This project is a glorious mess of modern OpenGL (`VAO`s, shaders), ancient immediate mode calls (`glBegin`/`glEnd` for the wireframe), and questionable global state management. It was written with the primary goal of "make it work," followed closely by "make it work faster." Readability and best practices were, at best, a tertiary concern. It's a learning project, and sometimes learning is messy.

Enjoy the chaos!
