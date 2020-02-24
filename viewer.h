#pragma once

#include <QtCore>
#include <QApplication>
#include <QWidget>
#include <QImage>
#include <QPainter>

#include "compute.h"

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

            fc.computeView();
            fc.fillImageData(image.bits());

            p.drawImage(rect(), image, image.rect());
        }

    private:

        int W;
        int H;
        QImage image;
        FractalCompute fc;
};