import os

def mainMenu():

    print("Sentiment Analysis")
    print("Nadia Rogriguez und Oscar Rivas")
    print("\n")
    print("########################")

    print("Optionen:\n")
    print("1. Englisch")
    print("2. Spanisch")
    print("3. Exit")

    while True:
        try:
            main_menu_selection = int(input("Enter option:"))
            #Englisch Option
            if main_menu_selection == 1:
                os.system('cls' if os.name =='nt' else 'clear')
                print("1. Bi")
                print("2. Multiclass")
                englisch_menu_selection == int(input("Enter option:"))
                if englisch_menu_selection == 1:
                    os.system('cls' if os.name =='nt' else 'clear')
                    englisch_menu_alg_selector = int(input("Select an Algorithm"))
                    print("1. MultinomialNB")
                    print("2. SVM")
                    print("3. Logistic Regression")
            else:
                break


"""def mainMenu():
    print("1. Do something good")
    print("2. Do something bad")
    print("3. Quit")
    while True:
        try:
            selection = int(input("Enter choice:"))
            if selection == 1:
                #good()
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("1. Do something good")
                print("2. Do something bad")
                selection_2 = int(input("Enter option:"))
                
                if selection_2 ==1:
                    good()
            elif selection ==2:
                bad()
            elif selection ==3:
                break
            
            else:
                print("Enter a valid option") 
                mainMenu()
        except ValueError:
            print ("Invalid option")

def good():
    print("Good")
    #anykey = input("Press some Button to return to the main Menu")
    #   mainMenu()

def bad():
    print("bad")
    anykey = input("Press some Button to return to the main Menu")
    mainMenu()
"""
mainMenu()