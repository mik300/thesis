/*
 * ${config.filename}
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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

#include "pmsis.h"
#include "pulp_nn_utils.h"
<%
act_prec = int(config.kernel.act_prec[0:2])
act_t = f"int{act_prec}_t"
def su(sgn):
    return 's' if sgn else 'u'
def u_(sgn):
    return '' if sgn else 'u'
def s_(sgn):
    return 's' if sgn else ''

pt_in = f"{u_(config.kernel.in_signed)}int8_t"
vt_in = f"v4{su(config.kernel.in_signed)}"
int_t_in = f"{u_(config.kernel.in_signed)}int32_t"
pt_out = f"{u_(config.kernel.out_signed)}int8_t"
sumdotp_fn = f"SumDotp{s_(config.kernel.in_signed)}4"
bex = f"bitext{u_(config.kernel.in_signed)}"
%>

void ${config.fn_name}(
                  ${pt_in} *pIn,
                  int8_t *pBias,
                  ${pt_out} *pOut,
                  int8_t *pWeight,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons)
{
%if config.kernel.in_data_t == 8:
    uint16_t dim_vec_in = dim_vec;
%elif config.kernel.in_data_t == 4:
    uint16_t dim_vec_in = dim_vec >> 1;
%elif config.kernel.in_data_t == 2:
    uint16_t dim_vec_in = dim_vec >> 2;
%endif
%if config.kernel.wt_data_t == 8:
    uint16_t dim_vec_wt = dim_vec;
%elif config.kernel.wt_data_t == 4:
    uint16_t dim_vec_wt = dim_vec >> 1;
%elif config.kernel.wt_data_t == 2:
    uint16_t dim_vec_wt = dim_vec >> 2;
%endif

    int core_id = pi_core_id();
    int Log2Core = log2(NUM_CORES);
    int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
    int start = min(chunk * core_id, num_o_neurons);
    int stop = min(start + chunk, num_o_neurons);

%if config.less_precision == 8:
    ${vt_in} vecA;
    v4s vecB;
    v4s vecB2;
%elif config.less_precision == 4:
    ${vt_in} vecA[2];
    v4s vecB[2];
    v4s vecB2[2];
%elif config.less_precision == 2:
    ${vt_in} vecA[4];
    v4s vecB[4];
    v4s vecB2[4];
%endif

    int32_t *pOutBuffer = (int32_t *) pOut + start;

    int lft_neurons = chunk & 0x01;
    int stop_even = stop - lft_neurons;
    int i;

    for(i=start; i<stop_even; i+=2)
    {
        int sum = 0;
        int sum2 = 0;
        if (pBias != NULL)
        {
          sum = *(int32_t *)(pBias + 4*i);
          sum2 = *(int32_t *)(pBias + 4*i + 4);
        }

        ${pt_in} *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);
        int8_t *pB2 = pB + dim_vec_wt;

%if config.less_precision == 8:
        for (int j=0; j<(dim_vec >> 2); j++)
%elif config.less_precision == 4:
        for (int j=0; j<(dim_vec >> 3); j++)
%elif config.less_precision == 2:
        for (int j=0; j<(dim_vec >> 4); j++)
%endif
        {
%if config.less_precision == 8:
               vecA = *((${vt_in}*)pA);
               vecB = *((v4s*)pB);
               vecB2 = *((v4s*)pB2);
             sum = ${sumdotp_fn}(vecA, vecB, sum);
             sum2 = ${sumdotp_fn}(vecA, vecB2, sum2);
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 8:
             vecA[0] = *((${vt_in}*)pA);
             pA+=4;
             vecA[1] = *((${vt_in}*)pA);
%else:
             ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
            vecB[0] = *((v4s*)pB);
            vecB2[0] = *((v4s*)pB2);
            pB+=4;
            pB2+=4;
            vecB[1] = *((v4s*)pB);
            vecB2[1] = *((v4s*)pB2);
%else:
            ${config.unpack_wt_fn}(pB,vecB);
            ${config.unpack_wt_fn}(pB2,vecB2);
%endif
                sum = ${sumdotp_fn}(vecA[0], vecB[0], sum);
                  sum = ${sumdotp_fn}(vecA[1], vecB[1], sum);
                  sum2 = ${sumdotp_fn}(vecA[0], vecB2[0], sum2);
                  sum2 = ${sumdotp_fn}(vecA[1], vecB2[1], sum2);
%elif config.less_precision == 2:
%if config.kernel.in_data_t == 8:
            vecA[0] = *((${vt_in}*)pA);
            pA+=4;
            vecA[1] = *((${vt_in}*)pA);
            pA+=4;
            vecA[2] = *((${vt_in}*)pA);
            pA+=4;
            vecA[3] = *((${vt_in}*)pA);
%elif config.kernel.in_data_t == 4:
                  ${config.unpack_in_fn}(pA,vecA);
                  pA+=4;
                  ${config.unpack_in_fn}(pA,vecA + 2);
%elif config.kernel.in_data_t == 2:
                  ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
            vecB[0] = *((v4s*)pB);
            vecB2[0] = *((v4s*)pB2);
            pB+=4;
            pB2+=4;
            vecB[1] = *((v4s*)pB);
                  vecB2[1] = *((v4s*)pB2);
            pB+=4;
            pB2+=4;
                  vecB[2] = *((v4s*)pB);
            vecB2[2] = *((v4s*)pB2);
            pB+=4;
            pB2+=4;
            vecB[3] = *((v4s*)pB);
                  vecB2[3] = *((v4s*)pB2);
%elif config.kernel.wt_data_t == 4:
              ${config.unpack_wt_fn}(pB,vecB);
                  ${config.unpack_wt_fn}(pB2,vecB2);
                  pB+=4;
                pB2+=4;
                  ${config.unpack_wt_fn}(pB,vecB + 2);
                  ${config.unpack_wt_fn}(pB2,vecB2 + 2);
%elif config.kernel.wt_data_t == 2:
            ${config.unpack_wt_fn}(pB,vecB);
            ${config.unpack_wt_fn}(pB2,vecB2);
%endif
                sum = ${sumdotp_fn}(vecA[0], vecB[0], sum);
                  sum = ${sumdotp_fn}(vecA[1], vecB[1], sum);
                  sum = ${sumdotp_fn}(vecA[2], vecB[2], sum);
                  sum = ${sumdotp_fn}(vecA[3], vecB[3], sum);
                  sum2 = ${sumdotp_fn}(vecA[0], vecB2[0], sum2);
                  sum2 = ${sumdotp_fn}(vecA[1], vecB2[1], sum2);
                  sum2 = ${sumdotp_fn}(vecA[2], vecB2[2], sum2);
                  sum2 = ${sumdotp_fn}(vecA[3], vecB2[3], sum2);
%endif
                  pA+=4;
                  pB+=4;
                  pB2+=4;
        }
%if config.less_precision == 2:
            uint16_t col_cnt = dim_vec & 0xf;
%elif config.less_precision == 4:
            uint16_t col_cnt = dim_vec & 0x7;
%elif config.less_precision == 8:
            uint16_t col_cnt = dim_vec & 0x3;
%endif
            while (col_cnt)
            {
%if config.less_precision == 2:
%if config.kernel.in_data_t == 2:
                  ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 0);
                  ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 2);
                  ${pt_in} inA3 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 4);
                  ${pt_in} inA4 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 6);
                  pA++;
%elif config.kernel.in_data_t == 4:
                  ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
                  ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
                  pA++;
                  ${pt_in} inA3 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
                  ${pt_in} inA4 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
                  pA++;
%elif config.kernel.in_data_t == 8:
                  ${pt_in} inA = *pA;
                  pA++;
                  ${pt_in} inA2 = *pA;
                  pA++;
                  ${pt_in} inA3 = *pA;
                  pA++;
                  ${pt_in} inA4 = *pA;
                  pA++;
%endif
%if config.kernel.wt_data_t == 2:
                  int8_t inB = (int8_t) bitext((int) *pB, 2, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 2, 2);
                  int8_t inB3 = (int8_t) bitext((int) *pB, 2, 4);
                  int8_t inB4 = (int8_t) bitext((int) *pB, 2, 6);
                  pB++;
                  int8_t inB5 = (int8_t) bitext((int) *pB2, 2, 0);
                  int8_t inB6 = (int8_t) bitext((int) *pB2, 2, 2);
                  int8_t inB7 = (int8_t) bitext((int) *pB2, 2, 4);
                  int8_t inB8 = (int8_t) bitext((int) *pB2, 2, 6);
                  pB2++;
%elif config.kernel.wt_data_t == 4:
                  int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
                  int8_t inB3 = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB4 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
                  int8_t inB5 = (int8_t) bitext((int) *pB2, 4, 0);
                  int8_t inB6 = (int8_t) bitext((int) *pB2, 4, 4);
                  pB2++;
                  int8_t inB7 = (int8_t) bitext((int) *pB2, 4, 0);
                  int8_t inB8 = (int8_t) bitext((int) *pB2, 4, 4);
                  pB2++;
%elif config.kernel.wt_data_t == 8:
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB2 = *pB;
                  pB++;
                  int8_t inB3 = *pB;
                  pB++;
                  int8_t inB4 = *pB;
                  pB++;
                  int8_t inB5 = *pB2;
                  pB2++;
                  int8_t inB6 = *pB2;
                  pB2++;
                  int8_t inB7 = *pB2;
                  pB2++;
                  int8_t inB8 = *pB2;
                  pB2++;
%endif
            sum += inA * inB;
            sum += inA2 * inB2;
            sum += inA3 * inB3;
            sum += inA4 * inB4;
            sum2 += inA * inB5;
            sum2 += inA2 * inB6;
            sum2 += inA3 * inB7;
            sum2 += inA4 * inB8;
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 4:
                  ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
                  ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
                  pA++;
%elif config.kernel.in_data_t == 8:
                  ${pt_in} inA = *pA;
                  pA++;
                  ${pt_in} inA2 = *pA;
                  pA++;
%endif
%if config.kernel.wt_data_t == 4:
                  int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
                  int8_t inB5 = (int8_t) bitext((int) *pB2, 4, 0);
                  int8_t inB6 = (int8_t) bitext((int) *pB2, 4, 4);
                  pB2++;
%elif config.kernel.wt_data_t == 8:
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB2 = *pB;
                  pB++;
                  int8_t inB5 = *pB2;
                  pB2++;
                  int8_t inB6 = *pB2;
                  pB2++;
%endif
            sum += inA * inB;
            sum += inA2 * inB2;
            sum2 += inA * inB5;
            sum2 += inA2 * inB6;
%elif config.less_precision == 8:
                  ${pt_in} inA = *pA;
                  pA++;
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB5 = *pB2;
                  pB2++;
                  sum += inA * inB;
                sum2 += inA * inB5;
%endif
                  col_cnt--;
            }
        *pOutBuffer = sum;
        pOutBuffer++;
        *pOutBuffer = sum2;
        pOutBuffer++;
    }
    if (lft_neurons && (stop - start) > 0)
    {
        int sum = 0;
        if (pBias != NULL)
        {
          sum = *(int32_t *)(pBias + 4*i);
        }

        ${pt_in} *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);

%if config.less_precision == 8:
        for (int j=0; j<(dim_vec >> 2); j++)
%elif config.less_precision == 4:
        for (int j=0; j<(dim_vec >> 3); j++)
%elif config.less_precision == 2:
        for (int j=0; j<(dim_vec >> 4); j++)
%endif
        {
%if config.less_precision == 8:
            vecA = *((${vt_in}*)pA);
            vecB = *((v4s*)pB);
            sum = ${sumdotp_fn}(vecA, vecB, sum);
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 8:
            vecA[0] = *((${vt_in}*)pA);
            pA+=4;
            vecA[1] = *((${vt_in}*)pA);
%else:
            ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
            vecB[0] = *((v4s*)pB);
            pB+=4;
            vecB[1] = *((v4s*)pB);
%else:
            ${config.unpack_wt_fn}(pB,vecB);
%endif
            sum = ${sumdotp_fn}(vecA[0], vecB[0], sum);
              sum = ${sumdotp_fn}(vecA[1], vecB[1], sum);
%elif config.less_precision == 2:
%if config.kernel.in_data_t == 8:
            vecA[0] = *((${vt_in}*)pA);
            pA+=4;
            vecA[1] = *((${vt_in}*)pA);
            pA+=4;
            vecA[2] = *((${vt_in}*)pA);
            pA+=4;
            vecA[3] = *((${vt_in}*)pA);
%elif config.kernel.in_data_t == 4:
              ${config.unpack_in_fn}(pA,vecA);
              pA+=4;
              ${config.unpack_in_fn}(pA,vecA + 2);
%elif config.kernel.in_data_t == 2:
              ${config.unpack_in_fn}(pA,vecA);
%endif
%if config.kernel.wt_data_t == 8:
           vecB[0] = *((v4s*)pB);
           pB+=4;
           vecB[1] = *((v4s*)pB);
           pB+=4;
             vecB[2] = *((v4s*)pB);
           pB+=4;
           vecB[3] = *((v4s*)pB);
%elif config.kernel.wt_data_t == 4:
           ${config.unpack_wt_fn}(pB,vecB);
           pB+=4;
               ${config.unpack_wt_fn}(pB,vecB + 2);
%elif config.kernel.wt_data_t == 2:
           ${config.unpack_wt_fn}(pB,vecB);
%endif
           sum = ${sumdotp_fn}(vecA[0], vecB[0], sum);
             sum = ${sumdotp_fn}(vecA[1], vecB[1], sum);
             sum = ${sumdotp_fn}(vecA[2], vecB[2], sum);
             sum = ${sumdotp_fn}(vecA[3], vecB[3], sum);
%endif
           pA+=4;
           pB+=4;
        }
%if config.less_precision == 2:
            uint16_t col_cnt = dim_vec & 0xf;
%elif config.less_precision == 4:
            uint16_t col_cnt = dim_vec & 0x7;
%elif config.less_precision == 8:
            uint16_t col_cnt = dim_vec & 0x3;
%endif
            while (col_cnt)
            {
%if config.less_precision == 2:
%if config.kernel.in_data_t == 2:
                  ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 0);
                  ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 2);
                  ${pt_in} inA3 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 4);
                  ${pt_in} inA4 = (${pt_in}) ${bex}((${int_t_in}) *pA, 2, 6);
                  pA++;
%elif config.kernel.in_data_t == 4:
                  ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
                  ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
                  pA++;
                  ${pt_in} inA3 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
                  ${pt_in} inA4 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
                  pA++;
%elif config.kernel.in_data_t == 8:
                  ${pt_in} inA = *pA;
                  pA++;
                  ${pt_in} inA2 = *pA;
                  pA++;
                  ${pt_in} inA3 = *pA;
                  pA++;
                  ${pt_in} inA4 = *pA;
                  pA++;
%endif
%if config.kernel.wt_data_t == 2:
                  int8_t inB = (int8_t) bitext((int) *pB, 2, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 2, 2);
                  int8_t inB3 = (int8_t) bitext((int) *pB, 2, 4);
                  int8_t inB4 = (int8_t) bitext((int) *pB, 2, 6);
                  pB++;
%elif config.kernel.wt_data_t == 4:
                  int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
                  int8_t inB3 = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB4 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
%elif config.kernel.wt_data_t == 8:
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB2 = *pB;
                  pB++;
                  int8_t inB3 = *pB;
                  pB++;
                  int8_t inB4 = *pB;
                  pB++;
%endif
            sum += inA * inB;
            sum += inA2 * inB2;
            sum += inA3 * inB3;
            sum += inA4 * inB4;
%elif config.less_precision == 4:
%if config.kernel.in_data_t == 4:
                  ${pt_in} inA = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 0);
                  ${pt_in} inA2 = (${pt_in}) ${bex}((${int_t_in}) *pA, 4, 4);
                  pA++;
%elif config.kernel.in_data_t == 8:
                  ${pt_in} inA = *pA;
                  pA++;
                  ${pt_in} inA2 = *pA;
                  pA++;
%endif
%if config.kernel.wt_data_t == 4:
                  int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
                  int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
                  pB++;
%elif config.kernel.wt_data_t == 8:
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB2 = *pB;
                  pB++;
%endif
            sum += inA * inB;
            sum += inA2 * inB2;
%elif config.less_precision == 8:
                  ${pt_in} inA = *pA;
                  pA++;
                  int8_t inB = *pB;
                  pB++;
                  sum += inA * inB;
%endif
                  col_cnt--;
            }
        *pOutBuffer = sum;
        pOutBuffer++;
    }
    pi_cl_team_barrier(0);
}
