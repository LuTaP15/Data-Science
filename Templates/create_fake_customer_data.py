"""
Documentation for Generate Fake Customer Data

This program is used to generate fake customer data for a fictional company.
The program generates a single row of data for each customer, which includes their name, city, postcode, age,
purchased item, price, and payment method.

Inputs:
N_CUSTOMERS - Number of customers to generate data for

Outputs:
Pandas dataframe containing fake customer data
"""

import pandas as pd
from faker import Faker
import random

# Parameters
N_CUSTOMERS = 10


# Generate fake customer data
def generate_customer_data():
    customer_data = []
    # Generate customer name
    fake = Faker('de_DE')
    customer_data.append(fake.first_name() + ' ' + fake.last_name())

    # Generate customer city
    customer_data.append(fake.city())
    # Generate customer postcode
    customer_data.append(fake.postcode())

    # Generate customer age
    customer_data.append(random.randint(10, 100))

    # Generate customer purchase item
    customer_data.append(random.choice(['shoes', 'shirt', 'pants', 'hat', 'socks', 'dress', 'jacket', 'scarf',
                                        'gloves', 'tie', 'belt', 'sunglasses', 'wallet', 'watch', 'jewelry',
                                        'boots', 'sandals', 'flip-flops', 'bag', 'backpack', 'umbrella',
                                        'keys', 'phone', 'toys', 'games', 'books', 'notebook', 'pen', 'pencil',
                                        'eraser', 'crayons', 'markers']))

    # Generate customer purchase price
    customer_data.append(str(random.randint(1, 100)) + '.' + str(random.randint(0, 99)) + ' €')

    # Generate customer payment method
    customer_data.append(random.choice(['Cash', 'EC Card', 'Credit Card']))

    return customer_data


if __name__ == "__main__":
    # Generate fake customer data and store in a Pandas dataframe
    df = pd.DataFrame(columns=['Name', 'City', 'Postcode', 'Age', 'Item', 'Price', 'Way_of_Payment'])
    for i in range(N_CUSTOMERS):
        customer_data = generate_customer_data()
        df.loc[i] = customer_data

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)