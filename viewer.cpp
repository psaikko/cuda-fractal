#include "viewer.h"

#include <sstream>
#include <chrono>
#include <QPainter>
#include <QMouseEvent>

using namespace std;
using namespace std::chrono;

Viewer::Viewer(int buffer_W, int buffer_H, QWidget *parent) : 
    QWidget(parent), 
    W(buffer_W), 
    H(buffer_H),
    image(W, H, QImage::Format::Format_RGBA8888),
    fc(W, H),
    hue_offset(0),
    hue_update(0.2),
    view_r_min(-2),
    view_r_max(1),
    view_i_min(-1),
    view_i_max(1),
    view_target_r(-0.5),
    view_target_i(0),
    zooming(0)
{
    memset(image.bits(), 0xff, W*H*4);
}

void Viewer::update() {
    // Update hue values
    hue_offset += hue_update;
    if (hue_offset > 255) hue_offset -= 255;
    if (hue_offset < -255) hue_offset += 255;

    // Parameters for zoom and scroll speed
    double zoom_factor;
    if (zooming == 1)
        zoom_factor = 0.98;
    else if (zooming == -1)
        zoom_factor = 1.0 / 0.98;
    else
        zoom_factor = 1.0;
    double shift_factor = 0.97;

    // Compute view center and dimensions
    double r_width = view_r_max - view_r_min;
    double i_height = view_i_max - view_i_min;

    double view_center_r = view_r_min + r_width / 2;
    double view_center_i = view_i_min + i_height / 2;

    // Re-center view on mouse with zoom applied;
    double target_width = r_width * zoom_factor;
    double target_height = i_height * zoom_factor;

    double shift_to_r = view_center_r * (shift_factor) +
                        view_target_r * (1 - shift_factor); 
    double shift_to_i = view_center_i * (shift_factor) +
                        view_target_i * (1 - shift_factor);

    view_r_min = shift_to_r - target_width / 2;
    view_r_max = shift_to_r + target_width / 2;
    view_i_min = shift_to_i - target_height / 2;
    view_i_max = shift_to_i + target_height / 2;

    QWidget::update(rect());
}

void Viewer::paintEvent(QPaintEvent *) {

    auto start_t = high_resolution_clock::now();

    QPainter p(this);
    p.setRenderHint(QPainter::SmoothPixmapTransform);

    fc.computeView(view_r_min, view_r_max, view_i_min, view_i_max, hue_offset);
    const unsigned char * data = fc.getData();

    auto mid_t = high_resolution_clock::now();

    unsigned char * imdata = image.bits();
    memcpy(imdata, data, 4*W*H);
    p.drawImage(rect(), image, image.rect());

    auto end_t = high_resolution_clock::now();
    auto cuda_ms = duration_cast<milliseconds>(mid_t - start_t);
    auto qt_ms = duration_cast<milliseconds>(end_t - mid_t);

    stringstream ss;
    ss << "[CUDA: " << cuda_ms.count() << " ms] [Qt: " << qt_ms.count() << "ms]";
    setWindowTitle(ss.str().c_str()); 
}

void Viewer::mouseMoveEvent(QMouseEvent *event) {
    double fx = double(event->x()) / double(width());
    double fy = double(event->y()) / double(height());

    // Compute new center
    double r_width = view_r_max - view_r_min;
    double i_height = view_i_max - view_i_min;
    view_target_r = view_r_min + r_width * fx;
    view_target_i = view_i_min + i_height * fy;
}

void Viewer::mousePressEvent(QMouseEvent *event) {
    if (event->button() & Qt::LeftButton)
        zooming = 1;
    if (event->button() & Qt::RightButton)
        zooming = -1;
}

void Viewer::mouseReleaseEvent(QMouseEvent *) {
    zooming = 0;
}