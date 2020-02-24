#include <QtCore>
#include <QApplication>
#include <QWidget>

#include <iostream>
#include "viewer.h"

using namespace std;

int main(int argc, char ** argv) {
    QApplication app(argc, argv);
    Viewer window(100,100);
    window.resize(100,100);
    window.setWindowTitle("test");
    window.show();

    return app.exec();
}