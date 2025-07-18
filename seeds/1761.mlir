module {
  func.func @main(%arg0: tensor<65x31x100x44x65xi16>, %arg1: tensor<65x31x100x1x1xi16>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<50x7x48x11xi32>, %arg5: tensor<1x7x1x1xi32>, %arg6: tensor<21x43x55x38xi1>) -> (tensor<i1>, tensor<1x8x1x10x1xi16>, tensor<50x7x48x11xi32>, tensor<1x43x1x38xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<65x31x100x44x65xi16>, tensor<65x31x100x1x1xi16>) -> tensor<65x31x100x44x65xi16>
    %s_1_start = tosa.const_shape {values = dense<[ 53, 23, 47, 20, 49 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_1_size = tosa.const_shape {values = dense<[ 1, 8, 10, 1, 1 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<65x31x100x44x65xi16>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<1x8x10x1x1xi16>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<1x8x10x1x1xi16>, tensor<1x8x10x1x1xi16>) -> tensor<1x8x10x1x1xi16>
    %3 = tosa.greater_equal %arg2, %arg3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %4 = tosa.intdiv %arg4, %arg5 : (tensor<50x7x48x11xi32>, tensor<1x7x1x1xi32>) -> tensor<50x7x48x11xi32>
    %5 = tosa.clz %3 : (tensor<i1>) -> tensor<i1>
    %6 = "tosa.const"() {values = dense<[3, 1, 0, 2, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
    %7 = tosa.transpose %2 {perms = array<i32: 0, 1, 3, 2, 4>} : (tensor<1x8x10x1x1xi16>) -> tensor<1x8x1x10x1xi16>
    %8 = tosa.reduce_all %arg6 {axis = 2 : i32} : (tensor<21x43x55x38xi1>) -> tensor<21x43x1x38xi1>
    %9 = tosa.minimum %4, %4 : (tensor<50x7x48x11xi32>, tensor<50x7x48x11xi32>) -> tensor<50x7x48x11xi32>
    %10 = tosa.reduce_min %8 {axis = 2 : i32} : (tensor<21x43x1x38xi1>) -> tensor<21x43x1x38xi1>
    %11 = tosa.reduce_max %10 {axis = 0 : i32} : (tensor<21x43x1x38xi1>) -> tensor<1x43x1x38xi1>
    return %5, %7, %9, %11 : tensor<i1>, tensor<1x8x1x10x1xi16>, tensor<50x7x48x11xi32>, tensor<1x43x1x38xi1>
  }
}
