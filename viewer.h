#pragma once

#include <QtCore>
#include <QApplication>
#include <QWidget>
#include <QImage>

#include "fractalcompute.h"

class Viewer : public QWidget {
    public:
        Viewer(int buffer_W, int buffer_H, QWidget *parent=0);

        void update();

        void paintEvent(QPaintEvent *);

    private:

        int W;
        int H;
        
        QImage image;
        FractalCompute fc;

        float hue_offset;
        float hue_update;
        int hue_begin;
        int hue_end;
};