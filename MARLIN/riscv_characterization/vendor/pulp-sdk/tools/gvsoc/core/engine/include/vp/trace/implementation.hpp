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


#ifndef __VP_TRACE_IMPLEMENTATION_HPP__
#define __VP_TRACE_IMPLEMENTATION_HPP__

#include "vp/vp_data.hpp"
#include "vp/trace/trace_engine.hpp"

namespace vp {

  inline trace_engine *component_trace::get_trace_manager()
  {
    if (this->trace_manager == NULL)
    {
      this->trace_manager = (trace_engine *)this->top.get_service("trace");
    }

    return this->trace_manager;
  }

  inline void vp::trace::event(uint8_t *value)
  {
  #ifdef VP_TRACE_ACTIVE
    if (is_event_active)
    {
      trace_manager->dump_event(this, comp->get_time(), value, bytes);
    }
  #endif
  }

  inline void vp::trace::event_pulse(int64_t duration, uint8_t *pulse_value, uint8_t *background_value)
  {
  #ifdef VP_TRACE_ACTIVE
    if (is_event_active)
    {
      trace_manager->dump_event_pulse(this, comp->get_time(), comp->get_clock()->get_time() + duration, pulse_value, background_value, bytes);
    }   
  #endif
  }

  inline void vp::trace::event_string(std::string value)
  {
  #ifdef VP_TRACE_ACTIVE
    if (is_event_active)
    {
      trace_manager->dump_event_string(this, comp->get_time(), (uint8_t *)value.c_str(), value.length() + 1);
    }   
  #endif
  }

  inline void vp::trace::event_real(double value)
  {
  #ifdef VP_TRACE_ACTIVE
    if (is_event_active)
    {
      trace_manager->dump_event(this, comp->get_time(), (uint8_t *)&value, 8);
    }  	
  #endif
  }

  inline void vp::trace::event_real_pulse(int64_t duration, double pulse_value, double background_value)
  {
  #ifdef VP_TRACE_ACTIVE
    if (is_event_active)
    {
      trace_manager->dump_event_pulse(this, comp->get_time(), comp->get_time() + duration, (uint8_t *)&pulse_value, (uint8_t *)&background_value, 8);
    }   
  #endif
  }

  inline void vp::trace::event_real_delayed(double value)
  {
  #ifdef VP_TRACE_ACTIVE
    if (is_event_active)
    {
      trace_manager->dump_event_delayed(this, comp->get_time(), (uint8_t *)&value, 8);
    }   
  #endif
  }


  inline void vp::trace::user_msg(const char *fmt, ...) {
    #if 0
    fprintf(trace_file, "%ld: %ld: [\033[34m%-*.*s\033[0m] ", comp->get_clock()->get_time(), comp->get_clock()->get_cycles(), max_trace_len, max_trace_len, comp->get_path());
    va_list ap;
    va_start(ap, fmt);
    if (vfprintf(trace_file, format, ap) < 0) {}
    va_end(ap);  
    #endif
  }

  inline void vp::trace::fatal(const char *fmt, ...)
  {
    dump_fatal_header();
    va_list ap;
    va_start(ap, fmt);
    if (vfprintf(this->trace_file, fmt, ap) < 0) {}
    va_end(ap);
    exit(1);
  }

  inline void vp::trace::warning(const char *fmt, ...) {
  #ifdef VP_TRACE_ACTIVE
  	if (is_active && comp->traces.get_trace_manager()->get_trace_level() >= this->level)
    {
      dump_warning_header();
      va_list ap;
      va_start(ap, fmt);
      if (vfprintf(this->trace_file, fmt, ap) < 0) {}
      va_end(ap);


      if (comp->traces.get_trace_manager()->get_werror())
      {
        exit(1);
      }
    }
  #endif
  }

  inline void vp::trace::warning(vp::trace::warning_type_e type, const char *fmt, ...) {
  #ifdef VP_TRACE_ACTIVE
  	if (is_active && comp->traces.get_trace_manager()->get_trace_level() >= vp::trace::LEVEL_WARNING)
    {
      if (comp->traces.get_trace_manager()->is_warning_active(type))
      {
        dump_warning_header();
        va_list ap;
        va_start(ap, fmt);
        if (vfprintf(this->trace_file, fmt, ap) < 0) {}
        va_end(ap);

        if (comp->traces.get_trace_manager()->get_werror())
        {
          exit(1);
        }
      }
    }
  #endif
  }

  inline void vp::trace::msg(const char *fmt, ...) 
  {
  #ifdef VP_TRACE_ACTIVE
  	if (is_active && comp->traces.get_trace_manager()->get_trace_level() >= this->level)
    {
      dump_header();
      va_list ap;
      va_start(ap, fmt);
      if (vfprintf(this->trace_file, fmt, ap) < 0) {}
      va_end(ap);  
    }
  #endif
  }

  inline void vp::trace::msg(int level, const char *fmt, ...) 
  {
  #ifdef VP_TRACE_ACTIVE
    if (is_active && comp->traces.get_trace_manager()->get_trace_level() >= level)
    {
      dump_header();
      if (level == vp::trace::LEVEL_ERROR)
      {
        fprintf(this->trace_file, "\033[31m");
      }
      else if (level == vp::trace::LEVEL_WARNING)
      {
        fprintf(this->trace_file, "\033[33m");
      }
      va_list ap;
      va_start(ap, fmt);
      if (vfprintf(this->trace_file, fmt, ap) < 0) {}
      va_end(ap);  
      if (level == vp::trace::LEVEL_ERROR || level == vp::trace::LEVEL_WARNING)
      {
        fprintf(this->trace_file, "\033[0m");
      }
    }
  #endif
  }


};

#endif
