module {
  func.func @main(%arg0: tensor<60x94x75x61xi1>, %arg1: tensor<76x55x47x69x19xi32>, %arg2: tensor<76x1x1x69x19xi32>, %arg3: tensor<21x56x34x30x90xf32>) -> (tensor<1x1x75x61xi1>, tensor<60x1x75x61xi1>, tensor<12x7x6x12x7xi1>, tensor<12x7x6x12x7xi32>, tensor<21x56x34x30x90xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<60x94x75x61xi1>) -> tensor<60x94x75x61xi1>
    %1 = tosa.reduce_all %0 {axis = 1 : i32} : (tensor<60x94x75x61xi1>) -> tensor<60x1x75x61xi1>
    %2 = tosa.intdiv %arg1, %arg2 : (tensor<76x55x47x69x19xi32>, tensor<76x1x1x69x19xi32>) -> tensor<76x55x47x69x19xi32>
    %3 = tosa.identity %1 : (tensor<60x1x75x61xi1>) -> tensor<60x1x75x61xi1>
    %4 = tosa.logical_not %3 : (tensor<60x1x75x61xi1>) -> tensor<60x1x75x61xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 24, 46, 38, 50, 12 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_5_size = tosa.const_shape {values = dense<[ 12, 7, 6, 12, 7 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %5 = tosa.slice %2, %s_5_start, %s_5_size : (tensor<76x55x47x69x19xi32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<12x7x6x12x7xi32>
    %6 = tosa.reverse %4 {axis = 1 : i32} : (tensor<60x1x75x61xi1>) -> tensor<60x1x75x61xi1>
    %7 = tosa.reduce_all %6 {axis = 1 : i32} : (tensor<60x1x75x61xi1>) -> tensor<60x1x75x61xi1>
    %8 = tosa.bitwise_not %7 : (tensor<60x1x75x61xi1>) -> tensor<60x1x75x61xi1>
    %9 = tosa.bitwise_not %8 : (tensor<60x1x75x61xi1>) -> tensor<60x1x75x61xi1>
    %10 = tosa.reduce_max %9 {axis = 0 : i32} : (tensor<60x1x75x61xi1>) -> tensor<1x1x75x61xi1>
    %11 = tosa.sub %10, %10 : (tensor<1x1x75x61xi1>, tensor<1x1x75x61xi1>) -> tensor<1x1x75x61xi1>
    %12 = tosa.greater_equal %5, %5 : (tensor<12x7x6x12x7xi32>, tensor<12x7x6x12x7xi32>) -> tensor<12x7x6x12x7xi1>
    %13 = tosa.reverse %8 {axis = 1 : i32} : (tensor<60x1x75x61xi1>) -> tensor<60x1x75x61xi1>
    %14 = tosa.sub %12, %12 : (tensor<12x7x6x12x7xi1>, tensor<12x7x6x12x7xi1>) -> tensor<12x7x6x12x7xi1>
    %15 = tosa.sub %5, %5 : (tensor<12x7x6x12x7xi32>, tensor<12x7x6x12x7xi32>) -> tensor<12x7x6x12x7xi32>
    %16 = tosa.floor %arg3 : (tensor<21x56x34x30x90xf32>) -> tensor<21x56x34x30x90xf32>
    return %11, %13, %14, %15, %16 : tensor<1x1x75x61xi1>, tensor<60x1x75x61xi1>, tensor<12x7x6x12x7xi1>, tensor<12x7x6x12x7xi32>, tensor<21x56x34x30x90xf32>
  }
}
