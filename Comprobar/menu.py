import os

def mainMenu():
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

mainMenu()