set -x

g++ -std=c++11 -Wall -Wunused-parameter -Wextra -pedantic -I ./ -O3 \
    framework/utils.cpp \
    framework/main.cpp \
    -o tinydnn

if [ $? -ne 0 ]; then
    echo "build failed"
    exit 1
fi

echo "build succ"
