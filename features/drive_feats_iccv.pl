#!/usr/bin/perl -w
use strict;
use File::Path;

# list of LFW names and ids
my $list = "data_iccv09/list.txt";

# the LAB color clusters file
my $cluster_file = "data_iccv09/lab_clusters";

## These directories should already exist on your system before running this code
# Can be found at http://vis-www.cs.umass.edu/lfw/ and http://vis-www.cs.umass.edu/lfw/part_labels/
#LFW Funnneled images
my $LFW_DIR= "data_iccv09/Color";
#Superpixels
my $SP_DIR= "data_iccv09/feat_superpixels";
#Ground Truth labels
my $LABEL_DIR="data_iccv09/GroundTruth";

## These directories should have been created earlier by generate_textures.m and generate_PB.m
my $TEXTON_DIR= "data_iccv09/parts_tex";
my $PB_DIR= "data_iccv09/parts_pb";

## These directories will be generated 
# Superpixel features
my $FEATURES_DIR= "data_iccv09/parts_spseg_features";
# Alternative representation of superpixels
my $SPMAT_DIR="data_iccv09/parts_superpixels_mat";
# Alternative representation of ground truth
my $GT_DIR="data_iccv09/parts_gt";

my $script = "generate_features";

print "Generating features...\n";

rmtree($FEATURES_DIR);
rmtree($GT_DIR);
rmtree($SPMAT_DIR);
 
mkdir($FEATURES_DIR);
mkdir($GT_DIR);
mkdir($SPMAT_DIR);
     
my $cmd = "./$script $list $cluster_file $LFW_DIR $SP_DIR $FEATURES_DIR $LABEL_DIR $GT_DIR $SPMAT_DIR $TEXTON_DIR $PB_DIR";

print "$cmd\n\n";

system($cmd);
