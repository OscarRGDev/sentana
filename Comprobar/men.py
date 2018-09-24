import os

def menu():
	"""
	Función que limpia la pantalla y muestra nuevamente el menu
	"""
	os.system('clear') # NOTA para windows tienes que cambiar clear por cls
	print ("Selecciona una opción")
	print ("\t1 - ES")
	print ("\t2 - EN")
	print ("\t3 - Exit")


while True:
	# Mostramos el menu
	menu()
	LanguageSelection = input("inserta un numero valor >> ")

	if LanguageSelection=="1":
        print ("ES:")
        print ("\t1- Multiclass Classifier")
        print ("\t1- Binary Classifier")
        input("Has pulsado la opción 1...\npulsa una tecla para continuar")
    else:
        print ("ciao")
"""
	elif LanguageSelection=="2":
		print ("")
		input("Has pulsado la opción 2...\npulsa una tecla para continuar")

	elif LanguageSelection=="3":
		print ("")
		input("Has pulsado la opción 3...\npulsa una tecla para continuar")

	elif LanguageSelection=="9":
		break

	else:
		print ("")
		input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")


"""