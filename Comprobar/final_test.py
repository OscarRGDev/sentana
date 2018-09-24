from cursesmenu import *
from cursesmenu.items import *
import os

menu = CursesMenu("Sentiment Analysis", "Select an option:")

selection_menu_language = SelectionMenu(["ES","EN"])
selection_menu_ES = SelectionMenu(["Multiclass classifier", "Binary Classifier"])
selection_menu_ES_multi = SelectionMenu(["Logistic Regression", "SVM", "Multinomial NB"])

menu.append_item(selection_menu_language)
menu.append_item(selection_menu_ES)
menu.append_item(selection_menu_ES_multi)

menu.show()