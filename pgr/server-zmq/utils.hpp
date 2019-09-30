#include <iostream>
#include <chrono>
#include <ctime>


class DebugCerr 
{
    using clock_src = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<long int, std::nano>; 
    using millis = std::chrono::duration<long int, std::milli>; 
    using micros = std::chrono::duration<long int, std::micro>; 

    private:
        std::chrono::time_point<clock_src> now, last;
        duration delta;

    public:

        DebugCerr()
        {
            last = clock_src::now();
        }

        template<class Any>
        std::ostream& operator<<(const Any& s)
        {
            now = clock_src::now();
            delta = now - last;
            last = now;

            return std::cerr << "+" << std::chrono::duration_cast<millis>(delta).count() << " ms: " << s; 
        }
};
