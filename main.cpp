#include <QtCore>
#include <QApplication>
#include <QWidget>
#include <QTimer>
#include <QObject>

#include <iostream>
#include <chrono>
#include "viewer.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char ** argv) {
    QApplication app(argc, argv);
    Viewer window(1500,1000);
    window.resize(1500,1000);
    window.setWindowTitle("Fractal viewer");
    window.setMouseTracking(true);
    window.show();

    QTimer screen_update_timer;
    screen_update_timer.setInterval(16ms);
    QObject::connect(&screen_update_timer, &QTimer::timeout, &window, &Viewer::update);
    screen_update_timer.start();

    return app.exec();
}