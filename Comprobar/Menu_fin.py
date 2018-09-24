import os

def menu():
    print (10 * "-" , "Sentiment Analysis" , 10 * "-")
    print ("\t 1. ES")
    print ("\t 2. EN")
    print ("\t 5. Exit")
    print ( 38* "-")

loop = True
while loop:
    menu()
    choice = input("Select an option: ")
    os.system('cls' if os.name == 'nt' else 'clear')
    if choice == "1":
        print("Menu 1 has been selected")
    elif choice == "2":
        print("Menu 2 has been selected")
    else:
        loop = False