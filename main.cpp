#include <QtCore>
#include <QApplication>
#include <QWidget>

#include <iostream>
#include "compute.h"

using namespace std;

int main(int argc, char ** argv) {
    QApplication app(argc, argv);
    QWidget window;
    window.resize(100,100);
    window.setWindowTitle("test");
    window.show();

    float f = compute();
    cout << f << endl;

    return app.exec();
}