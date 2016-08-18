pub const VERTEX_SHADER_SUBSTRATE: &'static str = "
                    #version 140
                    uniform mat4 matrix;
                    uniform mat4 perspective;
                    in vec3 position;
                    in vec4 color;
                    out vec4 fl_color;
                    void main() {
                        gl_Position = perspective * matrix * vec4(position, 1.0);
                        fl_color = color;
                    }
                ";

pub const FRAGMENT_SHADER_SUBSTRATE: &'static str = "
                    #version 140
                    in vec4 fl_color;
                    out vec4 color;
                    void main() {
                        color = fl_color;
                    }
                ";

pub const VERTEX_SHADER_VERTEX: &'static str = "
                    #version 140
                    uniform mat4 matrix;
                    in vec3 position;
                    in vec4 color;
                    out vec4 fl_color;
                    void main() {
                        gl_Position = matrix * vec4(position, 1.0);
                        fl_color = color;
                    }
";

pub const FRAGMENT_SHADER_VERTEX: &'static str = "
                    #version 140
                    in vec4 fl_color;
                    out vec4 color;
                    void main() {
                        color = fl_color;
                    }
";
