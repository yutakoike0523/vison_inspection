Input: full images image2-image18. Preprocessing: grayscale, mean removal, then FFT.
Two variants were generated: no window and 2-D Hann window. DC exclusion: radius <= 4 FFT bins.
Peak rows are conjugate-symmetric pairs; fx/fy units are cycles per pixel.
Reconstructions preserve the original complex FFT coefficients (including phase).
Hann reconstructions are intentionally not divided by the window: division near the zero-valued edge is unstable;
they therefore display the expected tapered amplitude. Main visual interpretation uses the no-window reconstruction.
