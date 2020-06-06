message = "Hello Python world!"
print(message)

name = "Ada Lovelace"
print(name.upper())
print(name.lower())
print(name.title())

first_name = "ada"
last_name = "lovelace"
full_name = f"{first_name} {last_name}"
print(full_name)

famous_person = "Albert Einstein"
quote = '"A person who never made a mistake never tried anything new"'
print(f"{famous_person} once said {quote}")

def funkyFunc(x):
    z = x * 10
    x = x % 10
    z = z + x
    return z

print(funkyFunc(12))

for i in range(1,101):
	if i % 3 == 0 and i % 5 == 0:
		print ('FizzBuzz')
	elif i % 3 == 0:
		print ('Fizz')
	elif i % 5 == 0:
		print('Buzz')
	else:
		print(i)

for i in range(0, 10):
	print(i)


i = 8
def func2():
    i += 10
    print(i)

func2()