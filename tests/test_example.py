import pytest

def sum(x,y):
    return x+y
    
def test_sum():
    x=5
    y=7
    z=sum(x,y)
    expected=12
    assert z==expected