
# Defining the source files for pulp-runtime
PULP_APP = ews_sum
PULP_APP_SRCS = $(wildcard ../../src/*.c) main.c
PULP_INC_PATHS += -DSDK
PULP_CFLAGS = -Wall -pedantic -O2 -g -I $(CURDIR)/../../include

#################################### RULES #####################################

include $(PULP_SDK_HOME)/install/rules/pulp.mk

##################################### EOF ######################################
