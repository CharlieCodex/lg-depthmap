#version 330
uniform sampler2D Texture;
in vec2 v_text;
out vec4 f_color;
void main() {
    f_color = vec4(texture(Texture, v_text/400).rgb, 1.0);
}