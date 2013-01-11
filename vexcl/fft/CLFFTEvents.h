// Light encapsulation of OpenCL events
// Copyright 2011, Eric Bainville

#ifndef CLFFTEvents_h
#define CLFFTEvents_h

#include <vector>
#include "CLFFTErrors.h"

#define CLFFT_CHECK_EVENT(e) CLFFT_CHECK_STATUS((e).getStatus())

namespace clfft {

class Context;
class EventVector;

// Encapsulates one OpenCL event and the status of its creation.
class Event
{
public:

  // Constructor. Initialized to event 0 (invalid).
  inline Event() : mEvent(0), mStatus(CL_INVALID_EVENT) { }

  // Copy
  inline Event(const Event & e) : mEvent(0) { assign(e.mEvent); mStatus = e.mStatus; }

  // = operator
  inline Event & operator = (const Event & e) { assign(e.mEvent); mStatus = e.mStatus; return *this; }

  // Destructor, release the event.
  inline ~Event() { assign(0); }

  // Check if valid (non 0).
  inline bool isValid() const { return (mEvent != 0); }

  // Get OpenCL status returned by the function creating the event
  inline cl_int getStatus() const { return mStatus; }

private:

  // Constructor: takes ownership of E without incrementing reference count.
  // E may be 0. STATUS is the status returned by the function creating E.
  inline Event(cl_event e,cl_int status) : mEvent(e), mStatus(status) { }

  // Special case for E=0 (to return errors).
  inline explicit Event(cl_int status) : mEvent(0), mStatus(status) { }

  // Special case for STATUS=CL_SUCCESS
  inline explicit Event(cl_event e) : mEvent(e), mStatus(CL_SUCCESS) { }

  // Access event
  inline operator cl_event () const { return mEvent; }

  // Assign value E, release previous event if any, and retain E if not 0. E may be 0.
  inline void assign(cl_event e)
  {
    if (e == mEvent) return; // Nothing to do
    if (mEvent != 0) { clReleaseEvent(mEvent); mEvent = 0; }
    if (e != 0) { mEvent = e; clRetainEvent(mEvent); }
  }

  // Encapsulated event, friends only
  cl_event mEvent;
  // Status returned when the event was created (normally CL_SUCCESS if event is valid)
  cl_int mStatus;

  friend class clfft::Context;
  friend class clfft::EventVector;
}; // class Event

// Encapsulates one vector of valid OpenCL events
class EventVector
{
public:

  // Constructor. Initialize with the given events. Events are retained.
  inline EventVector() { }
  inline EventVector(Event & e1) { append(e1); }
  inline EventVector(Event & e1,Event & e2) { append(e1); append(e2); }
  inline EventVector(Event & e1,Event & e2,Event & e3) { append(e1); append(e2); append(e3); }

  // Copy constructor
  inline EventVector(const EventVector & v)
  {
    size_t n = v.mEvents.size();
    for (size_t i=0;i<n;i++)
    {
      cl_event e = v.mEvents[i];
      clRetainEvent(e);
      mEvents.push_back(e);
    }
  }

  // = operator
  inline EventVector & operator = (const EventVector & v)
  {
    if (&v != this)
    {
      clear();
      size_t n = v.mEvents.size();
      for (size_t i=0;i<n;i++)
      {
        cl_event e = v.mEvents[i];
        clRetainEvent(e);
        mEvents.push_back(e);
      }
    }
    return *this;
  }

  // Destructor. Release the events.
  ~EventVector() { clear(); }

  // Append one event to the vector. Ignore if invalid. Otherwise the event is retained.
  inline void append(Event & e)
  {
    cl_event ce = (cl_event)e;
    if (ce == 0) return; // Invalid, ignore
    clRetainEvent(ce);
    mEvents.push_back(ce);
  }

  // Clear the vector. Release the events.
  inline void clear()
  {
    size_t n = mEvents.size();
    for (size_t i=0;i<n;i++) clReleaseEvent(mEvents[i]);
    mEvents.clear();
  }

private:

  // Access
  inline cl_uint size() const { return (cl_uint)mEvents.size(); }
  inline const cl_event * events() const { if (mEvents.empty()) return 0; else return &(mEvents[0]); }

  // Encapsulated events, friends only
  std::vector<cl_event> mEvents;

  friend class clfft::Context;
};

} // namespace

#endif // #ifndef CLFFTEvents_h
