/*
 * Copyright (C) 2020 GreenWaves Technologies, SAS, ETH Zurich and
 *                    University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* 
 * Authors: Germain Haugou, GreenWaves Technologies (germain.haugou@greenwaves-technologies.com)
 */

#ifndef __VP_CLOCK_EVENT_HPP__
#define __VP_CLOCK_EVENT_HPP__

#include "vp/vp_data.hpp"

namespace vp {

  class clock_event;
  class component;
  class component_clock;
  class clock_engine;

  #define CLOCK_EVENT_PAYLOAD_SIZE 64
  #define CLOCK_EVENT_NB_ARGS 8
  #define CLOCK_EVENT_QUEUE_SIZE 32
  #define CLOCK_EVENT_QUEUE_MASK (CLOCK_EVENT_QUEUE_SIZE - 1)

  typedef void (clock_event_meth_t)(void *, clock_event *event);

  class clock_event
  {

    friend class clock_engine;

  public:

    clock_event(component_clock *comp, clock_event_meth_t *meth);

    clock_event(component_clock *comp, void *_this, clock_event_meth_t *meth);

    ~clock_event();

    inline int get_payload_size() { return CLOCK_EVENT_PAYLOAD_SIZE; }
    inline uint8_t *get_payload() { return payload; }

    inline int get_nb_args() { return CLOCK_EVENT_NB_ARGS; }
    inline void **get_args() { return args; }

    inline bool is_enqueued() { return enqueued; }
    inline void set_clock(clock_engine *clock) { this->clock = clock; }

    inline int64_t get_cycle();

    void exec() { this->meth(this->_this, this); }

    inline void enqueue(int64_t cycles=1);
    inline void cancel();

    inline void enable();
    inline void disable();

    inline void meth_set(void *_this, clock_event_meth_t *meth) { this->_this = _this; this->meth = meth; }

    inline void stall_cycle_inc(int64_t inc) { this->stall_cycle += inc; }
    inline void stall_cycle_set(int64_t value) { this->stall_cycle = value; }
    inline int64_t stall_cycle_get() { return this->stall_cycle; }

  private:
    uint8_t payload[CLOCK_EVENT_PAYLOAD_SIZE];
    void *args[CLOCK_EVENT_NB_ARGS];
    component_clock *comp;
    void *_this;
    clock_event_meth_t *meth;
    clock_event *next;
    clock_event *prev;
    bool enqueued;
    int64_t cycle;
    int64_t stall_cycle;
    clock_engine *clock;
  };    

};

#endif
