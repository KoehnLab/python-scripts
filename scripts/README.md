# Python scripts

This directory is meant to contain executable Python scripts. Please ensure that you set the executable flag on your scripts and add the corresponding
shebang entry at the top of the file:
```python
#!/usr/bin/env python3

# Your code starts here
```
and to set the executable flag use
```bash
chmod +x your_file.py
```


## Best practices

For consistency, please use [snake\_case](https://en.wikipedia.org/wiki/Snake_case) for your file names. That means separating every word by an
underscore and only using lowercase letters. Examples: `my_script.py`, `helper.py`, `really_useful_script.py`

In case you were not aware, it is a good practice to use the following pattern in Python, which avoids cluttering the local namespace:
```
def main():
    # Put your code here

if __name__ == "__main__":
    main()
```

Example:
```python
# Defining functions or classes in the global namespace is fine
def say_hello():
    print("Hello")

def main():
    # Everything that is not a function or class definition, should only be performed
	# inside functions.
	# This main function will be the entry point of your script
    print("Starting program execution")

	say_hello()

if __name__ == "__main__":
    main()
```

