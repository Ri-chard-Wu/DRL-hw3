
pip install swig
pip install Box2D
pip install pygame
pip install pyglet==1.5.27
pip install shapely
apt-get install python-opengl

pip install setuptools==66
cp requirements.py /usr/local/lib/python3.8/dist-packages/wheel/vendored/packaging/requirements.py
pip install gym==0.20.0

apt-get update
apt-get install -y python-opengl
 
apt install -y xvfb
apt-get install -y libfontconfig1-dev

pip install opencv-python-headless
pip install pillow
# Run with '''xvfb-run -s "-screen 0 1400x900x24" python 111022533_hw3_train.py'''


# apt-get update && apt-get install -y vim


# -------------- HW3 環境安裝後，發生NoSuchDisplayException的error ----------------------------------------
# 我有遇到相同問題。我後來的解決辦法是先裝下面兩個 package:
 
# apt install -y xvfb
# apt-get install -y libfontconfig1-dev
 
# 然後用以下方式來執行 python 腳本:

#  xvfb-run -s "-screen 0 1400x900x24" python <Student_ID>_hw3_train.py