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

    protected:
        void paintEvent(QPaintEvent *);

        void mouseMoveEvent(QMouseEvent *event);

        void mousePressEvent(QMouseEvent *event);

        void mouseReleaseEvent(QMouseEvent *event);

    private:
        // Data buffer dimensions
        int W;
        int H;
        
        QImage image;
        FractalCompute fc;

        // Parameters for mapping iteration counts to hues
        float hue_offset;
        float hue_update;
        int hue_begin;
        int hue_end;

        // Bounds for real and imaginary components displayed
        float view_r_min;
        float view_r_max;
        float view_i_min;
        float view_i_max;

        float view_target_r;
        float view_target_i;

        bool zooming;
};