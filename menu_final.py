import os
os.system('cls' if os.name =='nt' else 'clear')
def menu():
    print("Sentiment Analysis")
    print ("\t1 - ES")
    print ("\t2 - EN")
    print ("\t3 - Exit")

while True:
    menu()
    language_choice = input("Enter an option: ")
    os.system('cls' if os.name =='nt' else 'clear')
    if language_choice =="1": 
        print ("ES Options:")
        print("\t1 - Multiclass classifier")
        print("\t2 - Binary classifier")
        ES_choiche = input("Enter an option: ")

        if ES_choiche == "1":
            os.system('cls' if os.name =='nt' else 'clear')
            print("Multiclass classifier. Select an Algorithm:")
            print("\t1 - Logistic Regression")
            print("\t2 - SVM")
            print("\t3 - MultinomialNB")
            print("\t4 - User input")
            ES_multiclass = input("Enter an option: ")

            if ES_multiclass == "1":
                os.system("python /Users/oscar/Desktop/Sentiment/ES/Multiclass/LogisticRegression.py")
                print("\n")

            elif ES_multiclass =="2":
                os.system("python /Users/oscar/Desktop/Sentiment/ES/Multiclass/SVM.py")
                print("\n")

            elif ES_multiclass == "3":
                os.system("python /Users/oscar/Desktop/Sentiment/ES/Multiclass/MultinomialNB.py")
                print("\n")

            elif ES_multiclass == "4":
                os.system("python /Users/oscar/Desktop/Sentiment/ES/Multiclass/User_input.py")
                print("\n")

        elif ES_choiche == "2":
            os.system('cls' if os.name =='nt' else 'clear')
            print("\tBinary classifier. Select an Algorithm:")
            print("\t1 - Logistic Regression")
            print("\t2 - SVM")
            print("\t3 - MultinomialNB")
            print("\t4 - User input")
            ES_bin = input("Enter an option: ")

            if ES_bin == "1":
                os.system("python /Users/oscar/Desktop/Sentiment/ES/Bin/LogisticRegression_bin.py")
                print("\n")

            elif ES_bin =="2":
                os.system("python /Users/oscar/Desktop/Sentiment/ES/Bin/SVM_bi.py")
                print("\n")

            elif ES_bin == "3":
                os.system("python /Users/oscar/Desktop/Sentiment/ES/Bin/MultinomialNB_Bi.py")
                print("\n")

            elif ES_bin == "4":
                os.system("python /Users/oscar/Desktop/Sentiment/ES/Bin/User_Input_bi.py")
                print("\n")
    
    if language_choice =="2": 
        print ("EN Options:")
        print("\t1 - Multiclass classifier")
        print("\t2 - Binary classifier")
        EN_choiche = input("Enter an option: ")

        if EN_choiche == "1":
            os.system('cls' if os.name =='nt' else 'clear')
            print("Multiclass classifier. Select an Algorithm:")
            print("\t1 - Logistic Regression")
            print("\t2 - SVM")
            print("\t3 - MultinomialNB")
            print("\t4 - User input")
            EN_multiclass = input("Enter an option: ")

            if EN_multiclass == "1":
                os.system("python /Users/oscar/Desktop/Sentiment/EN/Multiclass/LogisticRegression.py")
                print("\n")

            elif EN_multiclass =="2":
                os.system("python /Users/oscar/Desktop/Sentiment/EN/Multiclass/SVM.py")
                print("\n")

            elif EN_multiclass == "3":
                os.system("python /Users/oscar/Desktop/Sentiment/EN/Multiclass/MultinomialNB.py")
                print("\n")

            elif EN_multiclass == "4":
                os.system("python /Users/oscar/Desktop/Sentiment/EN/Multiclass/MultinomialNB.py")
                print("\n")

        elif EN_choiche == "2":
            os.system('cls' if os.name =='nt' else 'clear')
            print("\tBinary classifier. Select an Algorithm:")
            print("\t1 - Logistic Regression")
            print("\t2 - SVM")
            print("\t3 - MultinomialNB")
            print("\t4 - User input")
            EN_bin = input("Enter an option: ")

            if EN_bin == "1":
                os.system("python /Users/oscar/Desktop/Sentiment/EN/Bin/LogisticRegression_bin.py")
                print("\n")

            elif EN_bin =="2":
                os.system("python /Users/oscar/Desktop/Sentiment/EN/Bin/SVM_bi.py")
                print("\n")

            elif EN_bin == "3":
                os.system("python /Users/oscar/Desktop/Sentiment/EN/Bin/MultinomialNB_Bi.py")
                print("\n")

            elif EN_bin == "4":
                os.system("python /Users/oscar/Desktop/Sentiment/EN/Bin/User_Input_bi.py")
                print("\n")