#include <QtCore>
#include <QApplication>
#include <QWidget>

#include <iostream>
#include "viewer.h"

using namespace std;

int main(int argc, char ** argv) {
    QApplication app(argc, argv);
    Viewer window(1500,1000);
    window.resize(1500,1000);
    window.setWindowTitle("test");
    window.show();

    return app.exec();
}