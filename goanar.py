airline_reservation = {}

print("""" Reservtion menu option
 ------------------------------------
 1 Reserve a seat :
 2 cancel a Reservation :
 3 Exit the program:
 """)

choice = int(input("Enter an chioce: "))

while choice != 3:
  choice = int(input("Enter an chioce: "))
  if choice == 1: 
      row = int(input("Row (1-25): "))
      while row <= 0 or row > 25:
        print('Invalid entry')
        row = int(input("Row (1-25): "))
      col = input("column (A, B, C, D)")
      while not (col == 'A' or col =='B' or col=='C' or col=='D'):
        print('Invalid entry')
        col = input("column (A, B, C, D)")    
      seat = str(row) + col
      if seat not in airline_reservation:
          name = input("what is your name? ")
          airline_reservation[seat] = name
          print("seat",(seat),"han been reserved for",str(name))
          choice = 0
          col = 0
          row = 0
      elif seat in airline_reservation:
          print("That seat is taken by",airline_reservation[seat])
          row = 0
          col = 0
          choice = 0
  elif choice == 2: 
      seat = (input("Enter a seat number: "))
      if seat in airline_reservation:
         del airline_reservation[(seat)]
         print("Reservation cancelled ")
         choice = 0
      else:
         print("no reservation found")
print("Goodbye!")