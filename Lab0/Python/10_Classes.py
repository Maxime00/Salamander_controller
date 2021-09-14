#!/usr/bin/env python3

"""This script introduces you to the usage of classes in Python. """

import cmc_pylog as pylog


#: Creating a new class

class Animal:
    """New class for animal class.
    """

    def __init__(self, name):
        """Initialization function with one input to set the name of
        the animal during initialization.  """
        super(Animal, self).__init__()
        self._name = None  # : Internal class attribute
        # _ before variable indicates that this variable should not
        # be used by the user directly.
        # - self is special keyword in classes that points to itself.
        # The name self is a convention and not a Python keyword.
        self.name = name

    # Use of properties to set and get class attributes
    @property
    def name(self):
        """Get the name of the animal  """
        return self._name

    @name.setter
    def name(self, animal_name):
        """
        Parameters
        ----------
        animal_name : <str>
            Name of the animal.
            If it is not a string then a warning is thrown
        Returns
        -------
        out : None
        """
        # Check if animal is a string or not
        if not isinstance(animal_name, str):
            pylog.warning(
                'You don\'t really know how to name your animals! ')
        # This also works and is useful for dealing with inheritance:
        if not isinstance(animal_name, str):
            pylog.warning(
                'You don\'t really know how to name your animals! (isinstance)')
        self._name = animal_name


# Use of this class
pylog.info('Create a new animal using the Animal class')

# Instantiate Animal object with a name
animal = Animal('scooby')

# Get the Class name and name of the animal created
pylog.info('I am {} and my name is {}'.format(type(animal),
                                              animal.name))

# Use of @properties allows you to have more control over the
# attributes of a class.
# In this example we throw a funny warning to the user if they
# create an animal with a name that is not a string.
new_animal = Animal(1)

# Class inheritance
# You can make use of existing classes and then extend them with more
# features.
# Since animal is very generic we can use it and extend it to create
# a cat class


class Cat(Animal):
    """Class Inheritance. Create Cat as a child of Animal class.
    You now get access to all the attributes and methods of animal class
    for free!
    """

    def __init__(self, cat_name):
        """You can use the name attribute from the Animal class. """
        super(Cat, self).__init__(cat_name)

    def say_meow(self):
        """
        Make the cat say its name and then meow.
        """

        pylog.info('My master calls me {} and meow!'.format(self.name))


# Create a new cat
chat = Cat('French')
# Make the cat say meow
# See how the Cat class inherits from Animal class to set and retrieve
# its name

# Methods of a class are accessed using the .dot operator
chat.say_meow()

# Check if the cat is an animal
print("Is the cat of type animal (type): {}".format(isinstance(chat, Animal)))
print("Is the cat an animal (isinstance): {}".format(isinstance(chat, Animal)))

