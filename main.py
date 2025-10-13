import sys
from PyQt5 import QtWidgets
from app_logic import ApplicationLogic

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = ApplicationLogic()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    import os
    if not os.path.exists('pos'):
        os.makedirs('pos')
    if not os.path.exists('neg'):
        os.makedirs('neg')
    main()