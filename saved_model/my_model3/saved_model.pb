??

??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??	
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@?*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:?*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
t
cond_1/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *!
shared_namecond_1/Adam/iter
m
$cond_1/Adam/iter/Read/ReadVariableOpReadVariableOpcond_1/Adam/iter*
_output_shapes
: *
dtype0	
x
cond_1/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_1
q
&cond_1/Adam/beta_1/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_1*
_output_shapes
: *
dtype0
x
cond_1/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_2
q
&cond_1/Adam/beta_2/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_2*
_output_shapes
: *
dtype0
v
cond_1/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namecond_1/Adam/decay
o
%cond_1/Adam/decay/Read/ReadVariableOpReadVariableOpcond_1/Adam/decay*
_output_shapes
: *
dtype0
?
cond_1/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namecond_1/Adam/learning_rate

-cond_1/Adam/learning_rate/Read/ReadVariableOpReadVariableOpcond_1/Adam/learning_rate*
_output_shapes
: *
dtype0
x
current_loss_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecurrent_loss_scale
q
&current_loss_scale/Read/ReadVariableOpReadVariableOpcurrent_loss_scale*
_output_shapes
: *
dtype0
h

good_stepsVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
good_steps
a
good_steps/Read/ReadVariableOpReadVariableOp
good_steps*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d/kernel/m
?
/cond_1/Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
?
cond_1/Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecond_1/Adam/conv2d/bias/m
?
-cond_1/Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
cond_1/Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namecond_1/Adam/conv2d_1/kernel/m
?
1cond_1/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namecond_1/Adam/conv2d_1/bias/m
?
/cond_1/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*.
shared_namecond_1/Adam/conv2d_2/kernel/m
?
1cond_1/Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
?
cond_1/Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv2d_2/bias/m
?
/cond_1/Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*.
shared_namecond_1/Adam/conv2d_3/kernel/m
?
1cond_1/Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_3/kernel/m*'
_output_shapes
:@?*
dtype0
?
cond_1/Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namecond_1/Adam/conv2d_3/bias/m
?
/cond_1/Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_3/bias/m*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_namecond_1/Adam/dense/kernel/m
?
.cond_1/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/kernel/m*
_output_shapes
:	?*
dtype0
?
cond_1/Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/dense/bias/m
?
,cond_1/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/bias/m*
_output_shapes
:*
dtype0
?
cond_1/Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_namecond_1/Adam/dense_1/kernel/m
?
0cond_1/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_1/kernel/m*
_output_shapes

:*
dtype0
?
cond_1/Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecond_1/Adam/dense_1/bias/m
?
.cond_1/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
cond_1/Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d/kernel/v
?
/cond_1/Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
?
cond_1/Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecond_1/Adam/conv2d/bias/v
?
-cond_1/Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
cond_1/Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namecond_1/Adam/conv2d_1/kernel/v
?
1cond_1/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namecond_1/Adam/conv2d_1/bias/v
?
/cond_1/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*.
shared_namecond_1/Adam/conv2d_2/kernel/v
?
1cond_1/Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
?
cond_1/Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv2d_2/bias/v
?
/cond_1/Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*.
shared_namecond_1/Adam/conv2d_3/kernel/v
?
1cond_1/Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_3/kernel/v*'
_output_shapes
:@?*
dtype0
?
cond_1/Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namecond_1/Adam/conv2d_3/bias/v
?
/cond_1/Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_3/bias/v*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_namecond_1/Adam/dense/kernel/v
?
.cond_1/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/kernel/v*
_output_shapes
:	?*
dtype0
?
cond_1/Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/dense/bias/v
?
,cond_1/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/bias/v*
_output_shapes
:*
dtype0
?
cond_1/Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_namecond_1/Adam/dense_1/kernel/v
?
0cond_1/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_1/kernel/v*
_output_shapes

:*
dtype0
?
cond_1/Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecond_1/Adam/dense_1/bias/v
?
.cond_1/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?[
value?[B?[ B?[
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
?

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses* 
?

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
?
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
?

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*
?

Mkernel
Nbias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses*
?
U
loss_scale
Vbase_optimizer
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem?m?m?m?)m?*m?7m?8m?Em?Fm?Mm?Nm?v?v?v?v?)v?*v?7v?8v?Ev?Fv?Mv?Nv?*
Z
0
1
2
3
)4
*5
76
87
E8
F9
M10
N11*
Z
0
1
2
3
)4
*5
76
87
E8
F9
M10
N11*
* 
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

aserving_default* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

E0
F1*

E0
F1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

M0
N1*

M0
N1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*
* 
* 
.
?current_loss_scale
?
good_steps*
* 
SM
VARIABLE_VALUEcond_1/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEcond_1/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEcond_1/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcond_1/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEcond_1/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
nh
VARIABLE_VALUEcurrent_loss_scaleBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
good_steps:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUE*
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
??
VARIABLE_VALUEcond_1/Adam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEcond_1/Adam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEcond_1/Adam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEcond_1/Adam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEcond_1/Adam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEcond_1/Adam/dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/dense_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEcond_1/Adam/dense_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEcond_1/Adam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEcond_1/Adam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEcond_1/Adam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEcond_1/Adam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEcond_1/Adam/dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcond_1/Adam/dense_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEcond_1/Adam/dense_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_conv2d_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_4923325
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp$cond_1/Adam/iter/Read/ReadVariableOp&cond_1/Adam/beta_1/Read/ReadVariableOp&cond_1/Adam/beta_2/Read/ReadVariableOp%cond_1/Adam/decay/Read/ReadVariableOp-cond_1/Adam/learning_rate/Read/ReadVariableOp&current_loss_scale/Read/ReadVariableOpgood_steps/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/cond_1/Adam/conv2d/kernel/m/Read/ReadVariableOp-cond_1/Adam/conv2d/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_1/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_1/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_2/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_2/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_3/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_3/bias/m/Read/ReadVariableOp.cond_1/Adam/dense/kernel/m/Read/ReadVariableOp,cond_1/Adam/dense/bias/m/Read/ReadVariableOp0cond_1/Adam/dense_1/kernel/m/Read/ReadVariableOp.cond_1/Adam/dense_1/bias/m/Read/ReadVariableOp/cond_1/Adam/conv2d/kernel/v/Read/ReadVariableOp-cond_1/Adam/conv2d/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_1/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_1/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_2/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_2/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_3/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_3/bias/v/Read/ReadVariableOp.cond_1/Adam/dense/kernel/v/Read/ReadVariableOp,cond_1/Adam/dense/bias/v/Read/ReadVariableOp0cond_1/Adam/dense_1/kernel/v/Read/ReadVariableOp.cond_1/Adam/dense_1/bias/v/Read/ReadVariableOpConst*<
Tin5
321		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_4923650
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biascond_1/Adam/itercond_1/Adam/beta_1cond_1/Adam/beta_2cond_1/Adam/decaycond_1/Adam/learning_ratecurrent_loss_scale
good_stepstotalcounttotal_1count_1cond_1/Adam/conv2d/kernel/mcond_1/Adam/conv2d/bias/mcond_1/Adam/conv2d_1/kernel/mcond_1/Adam/conv2d_1/bias/mcond_1/Adam/conv2d_2/kernel/mcond_1/Adam/conv2d_2/bias/mcond_1/Adam/conv2d_3/kernel/mcond_1/Adam/conv2d_3/bias/mcond_1/Adam/dense/kernel/mcond_1/Adam/dense/bias/mcond_1/Adam/dense_1/kernel/mcond_1/Adam/dense_1/bias/mcond_1/Adam/conv2d/kernel/vcond_1/Adam/conv2d/bias/vcond_1/Adam/conv2d_1/kernel/vcond_1/Adam/conv2d_1/bias/vcond_1/Adam/conv2d_2/kernel/vcond_1/Adam/conv2d_2/bias/vcond_1/Adam/conv2d_3/kernel/vcond_1/Adam/conv2d_3/bias/vcond_1/Adam/dense/kernel/vcond_1/Adam/dense/bias/vcond_1/Adam/dense_1/kernel/vcond_1/Adam/dense_1/bias/v*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_4923801??
??
?
#__inference__traced_restore_4923801
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel: .
 assignvariableop_3_conv2d_1_bias: <
"assignvariableop_4_conv2d_2_kernel: @.
 assignvariableop_5_conv2d_2_bias:@=
"assignvariableop_6_conv2d_3_kernel:@?/
 assignvariableop_7_conv2d_3_bias:	?2
assignvariableop_8_dense_kernel:	?+
assignvariableop_9_dense_bias:4
"assignvariableop_10_dense_1_kernel:.
 assignvariableop_11_dense_1_bias:.
$assignvariableop_12_cond_1_adam_iter:	 0
&assignvariableop_13_cond_1_adam_beta_1: 0
&assignvariableop_14_cond_1_adam_beta_2: /
%assignvariableop_15_cond_1_adam_decay: 7
-assignvariableop_16_cond_1_adam_learning_rate: 0
&assignvariableop_17_current_loss_scale: (
assignvariableop_18_good_steps:	 #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: I
/assignvariableop_23_cond_1_adam_conv2d_kernel_m:;
-assignvariableop_24_cond_1_adam_conv2d_bias_m:K
1assignvariableop_25_cond_1_adam_conv2d_1_kernel_m: =
/assignvariableop_26_cond_1_adam_conv2d_1_bias_m: K
1assignvariableop_27_cond_1_adam_conv2d_2_kernel_m: @=
/assignvariableop_28_cond_1_adam_conv2d_2_bias_m:@L
1assignvariableop_29_cond_1_adam_conv2d_3_kernel_m:@?>
/assignvariableop_30_cond_1_adam_conv2d_3_bias_m:	?A
.assignvariableop_31_cond_1_adam_dense_kernel_m:	?:
,assignvariableop_32_cond_1_adam_dense_bias_m:B
0assignvariableop_33_cond_1_adam_dense_1_kernel_m:<
.assignvariableop_34_cond_1_adam_dense_1_bias_m:I
/assignvariableop_35_cond_1_adam_conv2d_kernel_v:;
-assignvariableop_36_cond_1_adam_conv2d_bias_v:K
1assignvariableop_37_cond_1_adam_conv2d_1_kernel_v: =
/assignvariableop_38_cond_1_adam_conv2d_1_bias_v: K
1assignvariableop_39_cond_1_adam_conv2d_2_kernel_v: @=
/assignvariableop_40_cond_1_adam_conv2d_2_bias_v:@L
1assignvariableop_41_cond_1_adam_conv2d_3_kernel_v:@?>
/assignvariableop_42_cond_1_adam_conv2d_3_bias_v:	?A
.assignvariableop_43_cond_1_adam_dense_kernel_v:	?:
,assignvariableop_44_cond_1_adam_dense_bias_v:B
0assignvariableop_45_cond_1_adam_dense_1_kernel_v:<
.assignvariableop_46_cond_1_adam_dense_1_bias_v:
identity_48??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*?
value?B?0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_cond_1_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp&assignvariableop_13_cond_1_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_cond_1_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp%assignvariableop_15_cond_1_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_cond_1_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp&assignvariableop_17_current_loss_scaleIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_good_stepsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_cond_1_adam_conv2d_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp-assignvariableop_24_cond_1_adam_conv2d_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp1assignvariableop_25_cond_1_adam_conv2d_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp/assignvariableop_26_cond_1_adam_conv2d_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp1assignvariableop_27_cond_1_adam_conv2d_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp/assignvariableop_28_cond_1_adam_conv2d_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_cond_1_adam_conv2d_3_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_cond_1_adam_conv2d_3_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_cond_1_adam_dense_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp,assignvariableop_32_cond_1_adam_dense_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_cond_1_adam_dense_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp.assignvariableop_34_cond_1_adam_dense_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp/assignvariableop_35_cond_1_adam_conv2d_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp-assignvariableop_36_cond_1_adam_conv2d_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp1assignvariableop_37_cond_1_adam_conv2d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp/assignvariableop_38_cond_1_adam_conv2d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp1assignvariableop_39_cond_1_adam_conv2d_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp/assignvariableop_40_cond_1_adam_conv2d_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp1assignvariableop_41_cond_1_adam_conv2d_3_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp/assignvariableop_42_cond_1_adam_conv2d_3_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp.assignvariableop_43_cond_1_adam_dense_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp,assignvariableop_44_cond_1_adam_dense_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp0assignvariableop_45_cond_1_adam_dense_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp.assignvariableop_46_cond_1_adam_dense_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_48IdentityIdentity_47:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_48Identity_48:output:0*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?*
?
G__inference_sequential_layer_call_and_return_conditional_losses_4922976

inputs(
conv2d_4922942:
conv2d_4922944:*
conv2d_1_4922947: 
conv2d_1_4922949: *
conv2d_2_4922953: @
conv2d_2_4922955:@+
conv2d_3_4922959:@?
conv2d_3_4922961:	? 
dense_4922965:	?
dense_4922967:!
dense_1_4922970:
dense_1_4922972:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCalld
conv2d/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:??????????
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d/Cast:y:0conv2d_4922942conv2d_4922944*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_4922704?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_4922947conv2d_1_4922949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_4922723?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4922668?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_4922953conv2d_2_4922955*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_4922743?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4922680?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_4922959conv2d_3_4922961*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_4922763?
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4922775?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4922965dense_4922967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4922789?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4922970dense_1_4922972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4922807w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_1_layer_call_fn_4923356

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_4922723w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????

 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4922668

inputs
identity?
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_4923369

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: ?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????

 X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

 i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????

 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_2_layer_call_fn_4923388

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_4922743w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?S
?
"__inference__wrapped_model_4922659
conv2d_inputJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource: A
3sequential_conv2d_1_biasadd_readvariableop_resource: L
2sequential_conv2d_2_conv2d_readvariableop_resource: @A
3sequential_conv2d_2_biasadd_readvariableop_resource:@M
2sequential_conv2d_3_conv2d_readvariableop_resource:@?B
3sequential_conv2d_3_biasadd_readvariableop_resource:	?B
/sequential_dense_matmul_readvariableop_resource:	?>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?*sequential/conv2d_3/BiasAdd/ReadVariableOp?)sequential/conv2d_3/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOpu
sequential/conv2d/CastCastconv2d_input*

DstT0*

SrcT0*/
_output_shapes
:??????????
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv2d/Conv2D/CastCast/sequential/conv2d/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:?
sequential/conv2d/Conv2DConv2Dsequential/conv2d/Cast:y:0!sequential/conv2d/Conv2D/Cast:y:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv2d/BiasAdd/CastCast0sequential/conv2d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:0"sequential/conv2d/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????|
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential/conv2d_1/Conv2D/CastCast1sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: ?
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:0#sequential/conv2d_1/Conv2D/Cast:y:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
 sequential/conv2d_1/BiasAdd/CastCast2sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: ?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:0$sequential/conv2d_1/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????

 ?
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 ?
 sequential/max_pooling2d/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*
T0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential/conv2d_2/Conv2D/CastCast1sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @?
sequential/conv2d_2/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:0#sequential/conv2d_2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
 sequential/conv2d_2/BiasAdd/CastCast2sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:0$sequential/conv2d_2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????@?
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential/conv2d_3/Conv2D/CastCast1sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@??
sequential/conv2d_3/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:0#sequential/conv2d_3/Conv2D/Cast:y:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
 sequential/conv2d_3/BiasAdd/CastCast2sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:??
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:0$sequential/conv2d_3/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:???????????
sequential/conv2d_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
sequential/flatten/ReshapeReshape&sequential/conv2d_3/Relu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential/dense/MatMul/CastCast.sequential/dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	??
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0 sequential/dense/MatMul/Cast:y:0*
T0*'
_output_shapes
:??????????
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense/BiasAdd/CastCast/sequential/dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0!sequential/dense/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:??????????
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/dense_1/MatMul/CastCast0sequential/dense_1/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:?
sequential/dense_1/MatMulMatMul!sequential/dense/BiasAdd:output:0"sequential/dense_1/MatMul/Cast:y:0*
T0*'
_output_shapes
:??????????
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_1/BiasAdd/CastCast1sequential/dense_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:0#sequential/dense_1/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:?????????r
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2X
*sequential/conv2d_3/BiasAdd/ReadVariableOp*sequential/conv2d_3/BiasAdd/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
?
B__inference_dense_layer_call_and_return_conditional_losses_4922789

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0k
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	?[
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_4922723

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: ?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????

 X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

 i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????

 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_3_layer_call_fn_4923420

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_4922763x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_4923401

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_dense_layer_call_fn_4923453

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4922789o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_dense_1_layer_call_and_return_conditional_losses_4923486

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0j
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:[
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_layer_call_fn_4923334

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_4922704w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_4923444

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_layer_call_fn_4923438

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4922775a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?
G__inference_sequential_layer_call_and_return_conditional_losses_4923108
conv2d_input(
conv2d_4923074:
conv2d_4923076:*
conv2d_1_4923079: 
conv2d_1_4923081: *
conv2d_2_4923085: @
conv2d_2_4923087:@+
conv2d_3_4923091:@?
conv2d_3_4923093:	? 
dense_4923097:	?
dense_4923099:!
dense_1_4923102:
dense_1_4923104:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCallj
conv2d/CastCastconv2d_input*

DstT0*

SrcT0*/
_output_shapes
:??????????
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d/Cast:y:0conv2d_4923074conv2d_4923076*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_4922704?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_4923079conv2d_1_4923081*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_4922723?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4922668?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_4923085conv2d_2_4923087*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_4922743?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4922680?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_4923091conv2d_3_4923093*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_4922763?
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4922775?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4923097dense_4923099*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4922789?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4923102dense_1_4923104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4922807w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_4922743

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
C__inference_conv2d_layer_call_and_return_conditional_losses_4922704

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_1_layer_call_fn_4923406

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4922680?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_dense_1_layer_call_and_return_conditional_losses_4922807

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0j
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:[
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_1_layer_call_fn_4923474

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4922807o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?	
G__inference_sequential_layer_call_and_return_conditional_losses_4923294

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@?7
(conv2d_3_biasadd_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpd
conv2d/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:??????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2D/CastCast$conv2d/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:?
conv2d/Conv2DConv2Dconv2d/Cast:y:0conv2d/Conv2D/Cast:y:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
conv2d/BiasAdd/CastCast%conv2d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0conv2d/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_1/Conv2D/CastCast&conv2d_1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: ?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0conv2d_1/Conv2D/Cast:y:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0z
conv2d_1/BiasAdd/CastCast'conv2d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: ?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0conv2d_1/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????

 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 ?
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
T0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_2/Conv2D/CastCast&conv2d_2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @?
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0conv2d_2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0z
conv2d_2/BiasAdd/CastCast'conv2d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0conv2d_2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_3/Conv2D/CastCast&conv2d_3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@??
conv2d_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0conv2d_3/Conv2D/Cast:y:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0{
conv2d_3/BiasAdd/CastCast'conv2d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:??
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0conv2d_3/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:??????????k
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
flatten/ReshapeReshapeconv2d_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0w
dense/MatMul/CastCast#dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	?y
dense/MatMulMatMulflatten/Reshape:output:0dense/MatMul/Cast:y:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0t
dense/BiasAdd/CastCast$dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:z
dense/BiasAddBiasAdddense/MatMul:product:0dense/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0z
dense_1/MatMul/CastCast%dense_1/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:{
dense_1/MatMulMatMuldense/BiasAdd:output:0dense_1/MatMul/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0x
dense_1/BiasAdd/CastCast&dense_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0dense_1/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_4922763

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0s
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@??
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_conv2d_layer_call_and_return_conditional_losses_4923347

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:o
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4922680

inputs
identity?
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?*
?
G__inference_sequential_layer_call_and_return_conditional_losses_4923070
conv2d_input(
conv2d_4923036:
conv2d_4923038:*
conv2d_1_4923041: 
conv2d_1_4923043: *
conv2d_2_4923047: @
conv2d_2_4923049:@+
conv2d_3_4923053:@?
conv2d_3_4923055:	? 
dense_4923059:	?
dense_4923061:!
dense_1_4923064:
dense_1_4923066:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCallj
conv2d/CastCastconv2d_input*

DstT0*

SrcT0*/
_output_shapes
:??????????
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d/Cast:y:0conv2d_4923036conv2d_4923038*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_4922704?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_4923041conv2d_1_4923043*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_4922723?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4922668?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_4923047conv2d_2_4923049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_4922743?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4922680?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_4923053conv2d_3_4923055*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_4922763?
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4922775?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4923059dense_4923061*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4922789?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4923064dense_1_4923066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4922807w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4923379

inputs
identity?
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?*
?
G__inference_sequential_layer_call_and_return_conditional_losses_4922814

inputs(
conv2d_4922705:
conv2d_4922707:*
conv2d_1_4922724: 
conv2d_1_4922726: *
conv2d_2_4922744: @
conv2d_2_4922746:@+
conv2d_3_4922764:@?
conv2d_3_4922766:	? 
dense_4922790:	?
dense_4922792:!
dense_1_4922808:
dense_1_4922810:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCalld
conv2d/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:??????????
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d/Cast:y:0conv2d_4922705conv2d_4922707*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_4922704?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_4922724conv2d_1_4922726*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_4922723?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4922668?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_4922744conv2d_2_4922746*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_4922743?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4922680?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_4922764conv2d_3_4922766*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_4922763?
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4922775?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4922790dense_4922792*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4922789?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4922808dense_1_4922810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4922807w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_4923325
conv2d_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_4922659o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
?
,__inference_sequential_layer_call_fn_4923032
conv2d_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4922976o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
?
,__inference_sequential_layer_call_fn_4923143

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4922814o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_4923433

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0s
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@??
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_4923172

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4922976o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_layer_call_fn_4923374

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4922668?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_layer_call_and_return_conditional_losses_4923465

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0k
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	?[
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_4922841
conv2d_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4922814o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?a
?
 __inference__traced_save_4923650
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop/
+savev2_cond_1_adam_iter_read_readvariableop	1
-savev2_cond_1_adam_beta_1_read_readvariableop1
-savev2_cond_1_adam_beta_2_read_readvariableop0
,savev2_cond_1_adam_decay_read_readvariableop8
4savev2_cond_1_adam_learning_rate_read_readvariableop1
-savev2_current_loss_scale_read_readvariableop)
%savev2_good_steps_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_cond_1_adam_conv2d_kernel_m_read_readvariableop8
4savev2_cond_1_adam_conv2d_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_1_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_1_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_2_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_2_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_3_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_3_bias_m_read_readvariableop9
5savev2_cond_1_adam_dense_kernel_m_read_readvariableop7
3savev2_cond_1_adam_dense_bias_m_read_readvariableop;
7savev2_cond_1_adam_dense_1_kernel_m_read_readvariableop9
5savev2_cond_1_adam_dense_1_bias_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_kernel_v_read_readvariableop8
4savev2_cond_1_adam_conv2d_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_1_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_1_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_2_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_2_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_3_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_3_bias_v_read_readvariableop9
5savev2_cond_1_adam_dense_kernel_v_read_readvariableop7
3savev2_cond_1_adam_dense_bias_v_read_readvariableop;
7savev2_cond_1_adam_dense_1_kernel_v_read_readvariableop9
5savev2_cond_1_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*?
value?B?0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop+savev2_cond_1_adam_iter_read_readvariableop-savev2_cond_1_adam_beta_1_read_readvariableop-savev2_cond_1_adam_beta_2_read_readvariableop,savev2_cond_1_adam_decay_read_readvariableop4savev2_cond_1_adam_learning_rate_read_readvariableop-savev2_current_loss_scale_read_readvariableop%savev2_good_steps_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_cond_1_adam_conv2d_kernel_m_read_readvariableop4savev2_cond_1_adam_conv2d_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_1_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_1_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_2_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_2_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_3_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_3_bias_m_read_readvariableop5savev2_cond_1_adam_dense_kernel_m_read_readvariableop3savev2_cond_1_adam_dense_bias_m_read_readvariableop7savev2_cond_1_adam_dense_1_kernel_m_read_readvariableop5savev2_cond_1_adam_dense_1_bias_m_read_readvariableop6savev2_cond_1_adam_conv2d_kernel_v_read_readvariableop4savev2_cond_1_adam_conv2d_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_1_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_1_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_2_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_2_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_3_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_3_bias_v_read_readvariableop5savev2_cond_1_adam_dense_kernel_v_read_readvariableop3savev2_cond_1_adam_dense_bias_v_read_readvariableop7savev2_cond_1_adam_dense_1_kernel_v_read_readvariableop5savev2_cond_1_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
220		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : @:@:@?:?:	?:::: : : : : : : : : : : ::: : : @:@:@?:?:	?:::::: : : @:@:@?:?:	?:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:%	!

_output_shapes
:	?: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
: @: )

_output_shapes
:@:-*)
'
_output_shapes
:@?:!+

_output_shapes	
:?:%,!

_output_shapes
:	?: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::0

_output_shapes
: 
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_4922775

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?E
?	
G__inference_sequential_layer_call_and_return_conditional_losses_4923233

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@?7
(conv2d_3_biasadd_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpd
conv2d/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:??????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2D/CastCast$conv2d/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:?
conv2d/Conv2DConv2Dconv2d/Cast:y:0conv2d/Conv2D/Cast:y:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
conv2d/BiasAdd/CastCast%conv2d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0conv2d/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_1/Conv2D/CastCast&conv2d_1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: ?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0conv2d_1/Conv2D/Cast:y:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0z
conv2d_1/BiasAdd/CastCast'conv2d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: ?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0conv2d_1/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????

 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 ?
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
T0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_2/Conv2D/CastCast&conv2d_2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @?
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0conv2d_2/Conv2D/Cast:y:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0z
conv2d_2/BiasAdd/CastCast'conv2d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0conv2d_2/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:?????????@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_3/Conv2D/CastCast&conv2d_3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@??
conv2d_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0conv2d_3/Conv2D/Cast:y:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0{
conv2d_3/BiasAdd/CastCast'conv2d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:??
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0conv2d_3/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:??????????k
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
flatten/ReshapeReshapeconv2d_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0w
dense/MatMul/CastCast#dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	?y
dense/MatMulMatMulflatten/Reshape:output:0dense/MatMul/Cast:y:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0t
dense/BiasAdd/CastCast$dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:z
dense/BiasAddBiasAdddense/MatMul:product:0dense/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0z
dense_1/MatMul/CastCast%dense_1/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:{
dense_1/MatMulMatMuldense/BiasAdd:output:0dense_1/MatMul/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0x
dense_1/BiasAdd/CastCast&dense_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0dense_1/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4923411

inputs
identity?
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
conv2d_input=
serving_default_conv2d_input:0?????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
?

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Mkernel
Nbias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
?
U
loss_scale
Vbase_optimizer
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem?m?m?m?)m?*m?7m?8m?Em?Fm?Mm?Nm?v?v?v?v?)v?*v?7v?8v?Ev?Fv?Mv?Nv?"
	optimizer
v
0
1
2
3
)4
*5
76
87
E8
F9
M10
N11"
trackable_list_wrapper
v
0
1
2
3
)4
*5
76
87
E8
F9
M10
N11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_sequential_layer_call_fn_4922841
,__inference_sequential_layer_call_fn_4923143
,__inference_sequential_layer_call_fn_4923172
,__inference_sequential_layer_call_fn_4923032?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_4923233
G__inference_sequential_layer_call_and_return_conditional_losses_4923294
G__inference_sequential_layer_call_and_return_conditional_losses_4923070
G__inference_sequential_layer_call_and_return_conditional_losses_4923108?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_4922659conv2d_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
aserving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_conv2d_layer_call_fn_4923334?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_layer_call_and_return_conditional_losses_4923347?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
):' 2conv2d_1/kernel
: 2conv2d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_1_layer_call_fn_4923356?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_4923369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_max_pooling2d_layer_call_fn_4923374?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4923379?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
):' @2conv2d_2/kernel
:@2conv2d_2/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_2_layer_call_fn_4923388?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_4923401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
?2?
1__inference_max_pooling2d_1_layer_call_fn_4923406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4923411?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
*:(@?2conv2d_3/kernel
:?2conv2d_3/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_3_layer_call_fn_4923420?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_4923433?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_flatten_layer_call_fn_4923438?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_layer_call_and_return_conditional_losses_4923444?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	?2dense/kernel
:2
dense/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_dense_layer_call_fn_4923453?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_4923465?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 :2dense_1/kernel
:2dense_1/bias
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_1_layer_call_fn_4923474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_4923486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
H
?current_loss_scale
?
good_steps"
_generic_user_object
"
_generic_user_object
:	 (2cond_1/Adam/iter
: (2cond_1/Adam/beta_1
: (2cond_1/Adam/beta_2
: (2cond_1/Adam/decay
#:! (2cond_1/Adam/learning_rate
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_4923325conv2d_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
: 2current_loss_scale
:	 2
good_steps
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
3:12cond_1/Adam/conv2d/kernel/m
%:#2cond_1/Adam/conv2d/bias/m
5:3 2cond_1/Adam/conv2d_1/kernel/m
':% 2cond_1/Adam/conv2d_1/bias/m
5:3 @2cond_1/Adam/conv2d_2/kernel/m
':%@2cond_1/Adam/conv2d_2/bias/m
6:4@?2cond_1/Adam/conv2d_3/kernel/m
(:&?2cond_1/Adam/conv2d_3/bias/m
+:)	?2cond_1/Adam/dense/kernel/m
$:"2cond_1/Adam/dense/bias/m
,:*2cond_1/Adam/dense_1/kernel/m
&:$2cond_1/Adam/dense_1/bias/m
3:12cond_1/Adam/conv2d/kernel/v
%:#2cond_1/Adam/conv2d/bias/v
5:3 2cond_1/Adam/conv2d_1/kernel/v
':% 2cond_1/Adam/conv2d_1/bias/v
5:3 @2cond_1/Adam/conv2d_2/kernel/v
':%@2cond_1/Adam/conv2d_2/bias/v
6:4@?2cond_1/Adam/conv2d_3/kernel/v
(:&?2cond_1/Adam/conv2d_3/bias/v
+:)	?2cond_1/Adam/dense/kernel/v
$:"2cond_1/Adam/dense/bias/v
,:*2cond_1/Adam/dense_1/kernel/v
&:$2cond_1/Adam/dense_1/bias/v?
"__inference__wrapped_model_4922659?)*78EFMN=?:
3?0
.?+
conv2d_input?????????
? "1?.
,
dense_1!?
dense_1??????????
E__inference_conv2d_1_layer_call_and_return_conditional_losses_4923369l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

 
? ?
*__inference_conv2d_1_layer_call_fn_4923356_7?4
-?*
(?%
inputs?????????
? " ??????????

 ?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_4923401l)*7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_2_layer_call_fn_4923388_)*7?4
-?*
(?%
inputs????????? 
? " ??????????@?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_4923433m787?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_3_layer_call_fn_4923420`787?4
-?*
(?%
inputs?????????@
? "!????????????
C__inference_conv2d_layer_call_and_return_conditional_losses_4923347l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_layer_call_fn_4923334_7?4
-?*
(?%
inputs?????????
? " ???????????
D__inference_dense_1_layer_call_and_return_conditional_losses_4923486\MN/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_1_layer_call_fn_4923474OMN/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_layer_call_and_return_conditional_losses_4923465]EF0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_layer_call_fn_4923453PEF0?-
&?#
!?
inputs??????????
? "???????????
D__inference_flatten_layer_call_and_return_conditional_losses_4923444b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
)__inference_flatten_layer_call_fn_4923438U8?5
.?+
)?&
inputs??????????
? "????????????
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4923411?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_1_layer_call_fn_4923406?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4923379?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_layer_call_fn_4923374?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_sequential_layer_call_and_return_conditional_losses_4923070|)*78EFMNE?B
;?8
.?+
conv2d_input?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_4923108|)*78EFMNE?B
;?8
.?+
conv2d_input?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_4923233v)*78EFMN??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_4923294v)*78EFMN??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_layer_call_fn_4922841o)*78EFMNE?B
;?8
.?+
conv2d_input?????????
p 

 
? "???????????
,__inference_sequential_layer_call_fn_4923032o)*78EFMNE?B
;?8
.?+
conv2d_input?????????
p

 
? "???????????
,__inference_sequential_layer_call_fn_4923143i)*78EFMN??<
5?2
(?%
inputs?????????
p 

 
? "???????????
,__inference_sequential_layer_call_fn_4923172i)*78EFMN??<
5?2
(?%
inputs?????????
p

 
? "???????????
%__inference_signature_wrapper_4923325?)*78EFMNM?J
? 
C?@
>
conv2d_input.?+
conv2d_input?????????"1?.
,
dense_1!?
dense_1?????????