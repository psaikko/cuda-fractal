#pragma once

#include <QtCore>
#include <QApplication>
#include <QWidget>
#include <QImage>
#include <QPainter>

#include "fractalcompute.h"

#define HUE_START 359
#define HUE_END 60

class Viewer : public QWidget {
    public:
        Viewer(int buffer_W, int buffer_H, QWidget *parent=0) : 
            QWidget(parent), 
            W(buffer_W), 
            H(buffer_H),
            image(W, H, QImage::Format::Format_RGBA8888),
            fc(W, H)
        {
            memset(image.bits(), 0xff, W*H*4);
        }

        void paintEvent(QPaintEvent *) {
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
                    int h = (HUE_END - HUE_START) * sqrt(data[i]) + HUE_START;
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

    private:

        int W;
        int H;
        QImage image;
        FractalCompute fc;
};