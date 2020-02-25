#include "viewer.h"

#include <QPainter>

Viewer::Viewer(int buffer_W, int buffer_H, QWidget *parent) : 
    QWidget(parent), 
    W(buffer_W), 
    H(buffer_H),
    image(W, H, QImage::Format::Format_RGBA8888),
    fc(W, H),
    hue_offset(0),
    hue_update(1),
    hue_begin(359),
    hue_end(60)
{
    memset(image.bits(), 0xff, W*H*4);
}

void Viewer::update() {
    hue_offset += hue_update;
    if (hue_offset > 360) hue_offset -= 360;
    if (hue_offset < -360) hue_offset += 360;
    QWidget::update(rect());
}

void Viewer::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::SmoothPixmapTransform);

    fc.computeView();
    const float * data = fc.getData();
    unsigned char * imdata = image.bits();

    for (int i = 0; i < W*H; ++i) {
        QColor c;
        if (abs(data[i] - 1.0) < 1e-4) {
            c = QColor::fromRgb(0,0,0);
        } else {
            int h = (hue_end - hue_begin) * sqrt(data[i]) + hue_begin + hue_offset;
            h %= 360;
            c = QColor::fromHsv(h,255,255);
        }
        int r, g, b;
        c.getRgb(&r, &g, &b);
        imdata[4*i + 0] = r;
        imdata[4*i + 1] = g;
        imdata[4*i + 2] = b;
        imdata[4*i + 3] = 255;
    }

    p.drawImage(rect(), image, image.rect());
}