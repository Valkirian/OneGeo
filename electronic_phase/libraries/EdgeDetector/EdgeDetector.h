#ifndef __EDGEDETECTOR_H__
#define __EDGEDETECTOR_H__


class EdgeDetector
{
 public:
    EdgeDetector();
    EdgeDetector(bool);

    void operator()(bool);

    // Updates the pin
    // Returns true if the state changed
    // Returns false if the state did not change
    bool update(bool);

    // Returns the updated pin state
    bool read();

    // Returns the falling pin state
    bool fell();

    // Returns the rising pin state
    bool rose();

 private:
    bool current;
    bool previous;
};

#endif //__EDGEDETECTOR_H__
