#version 140
uniform mat4 matrix;
in vec3 position;
in vec4 color;
out vec4 fl_color;
void main() {
    gl_Position = matrix * vec4(position, 1.0);
    fl_color = color;
}
