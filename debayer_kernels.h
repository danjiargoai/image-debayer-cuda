#pragma once

typedef unsigned char Pixel;
struct outPixel {
  unsigned char R, G, B;
};

extern "C" void hqDebayer(outPixel *odata, int iw, int ih);
extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp, StopWatchInterface *timer);
extern "C" void deleteTexture(void);
