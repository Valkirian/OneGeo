// Please read EdgeDetector2.h for information about the liscence and authors

#include "Arduino.h"
#include "EdgeDetector.h"


EdgeDetector::EdgeDetector()
    : current(false)
    , previous(false)
{}

EdgeDetector::EdgeDetector(bool value)
    : current(value)
    , previous(value)
{}

void EdgeDetector::operator()(bool value)
{
    current = previous = value;
}

bool EdgeDetector::update(bool value)
{
    previous = current;
    current = value;

    return (previous != current);
}

bool EdgeDetector::read()
{
    return current;
}

bool EdgeDetector::rose()
{
    bool flag = (( current ) && ( !previous ));
    if (flag) previous = current;
    return flag;
}

bool EdgeDetector::fell()
{
    bool flag = (( !current ) && ( previous ));
    if (flag) previous = current;
    return flag;
}
