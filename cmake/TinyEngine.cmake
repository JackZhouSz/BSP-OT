set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors -O")

# Définir les chemins d'inclusion et de bibliothèque
include_directories("$ENV{HOME}/.local/include")
link_directories("$ENV{HOME}/.local/lib")

# Détecter l'OS et ajuster les liens de bibliothèques
if(UNIX AND NOT APPLE)
    # Pour Linux
    set(TINYLINK -lpthread -lSDL2 -lSDL2_image -lSDL2_mixer -lSDL2_ttf -lGLEW -lboost_system -lboost_filesystem -lX11 -lGL)
elseif(APPLE)
    # Pour macOS
    set(TINYLINK -lpthread -lSDL2 -lSDL2_image -lSDL2_mixer -lSDL2_ttf -lGLEW -lboost_system -lboost_filesystem -framework OpenGL)
    include_directories("/opt/homebrew/include")
    link_directories("/opt/homebrew/lib")
endif()
