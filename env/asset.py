class Asset:
    def __init__(self, name, value, weighting, price, num_shares):
        self.__name = name
        self.__value = value
        self.__weighting = weighting
        self.__price = price
        self.__num_shares = num_shares
    
    def get_name(self):
        return self.__name
    
    def get_value(self):
        return self.__value
    
    def get_weighting(self):
        return self.__weighting

    def get_price(self):
        return self.__price
    
    def get_num_shares(self):
        return self.__num_shares
    
    def set_value(self, value):
        self.__value = value
    
    def set_weighting(self, weighting):
        self.__weighting = weighting

    def set_price(self, price):
        self.__price = price
    
    def set_num_shares(self, num_shares):
        self.__num_shares = num_shares