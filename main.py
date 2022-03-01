from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.screen import Screen
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.behaviors.magic_behavior import MagicBehavior

class MagicButton1(MagicBehavior, MDRaisedButton):
    pass

class LoginScreen(Screen):
    print("INITIALIZED: LOGIN SCREEN")

class MainStudent(Screen):
    print("INITIALIZED: student SCREEN")

class MainProfessor(Screen):
    print("INITIALIZED: prof SCREEN")

class MainAdmin(Screen):
    print("INITIALIZED: admin SCREEN")
    

# Main build class
class OECP(MDApp):
    def build(self):
        global sm
        self.load_kv('main.kv')
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name = 'kv_login'))
        sm.add_widget(MainStudent(name = 'kv_MainStudent'))
        sm.add_widget(MainProfessor(name = 'kv_MainProf'))
        sm.add_widget(MainAdmin(name = 'kv_MainAdmin'))
        print("INITIALIZED: SCREEN MANAGER AND SCREENS")
        return sm


if __name__ == "__main__":
    # # Kivy Initialization

    print("INITIALIZED: MAIN")
    Window.size = (600, 300)
    OECP().run()

