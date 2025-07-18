module {
  func.func @main(%arg0: tensor<71x21x82x34x5xi8>, %arg1: tensor<71x1x1x1x5xi8>, %arg2: tensor<17x74xi16>, %arg3: tensor<75x4x94xf32>, %arg4: tensor<56x91xi1>) -> (tensor<71x21x82x34x5xi1>, tensor<1x4x94xf32>, tensor<1x91xi1>, tensor<75x1x94xf32>, tensor<1x91xi1>, tensor<74xi32>, tensor<75x1x94xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<71x21x82x34x5xi8>, tensor<71x1x1x1x5xi8>) -> tensor<71x21x82x34x5xi8>
    %1 = tosa.argmax %arg2 {axis = 0 : i32} : (tensor<17x74xi16>) -> tensor<74xi32>
    %2 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 0>} : (tensor<74xi32>) -> tensor<74xi32>
    %4 = tosa.log %arg3 : (tensor<75x4x94xf32>) -> tensor<75x4x94xf32>
    %5 = tosa.equal %0, %0 : (tensor<71x21x82x34x5xi8>, tensor<71x21x82x34x5xi8>) -> tensor<71x21x82x34x5xi1>
    %6 = tosa.add %4, %4 : (tensor<75x4x94xf32>, tensor<75x4x94xf32>) -> tensor<75x4x94xf32>
    %7 = tosa.rsqrt %4 : (tensor<75x4x94xf32>) -> tensor<75x4x94xf32>
    %8 = tosa.concat %7, %4 {axis = 0 : i32} : (tensor<75x4x94xf32>, tensor<75x4x94xf32>) -> tensor<150x4x94xf32>
    %9 = tosa.logical_and %5, %5 : (tensor<71x21x82x34x5xi1>, tensor<71x21x82x34x5xi1>) -> tensor<71x21x82x34x5xi1>
    %10 = tosa.reduce_max %6 {axis = 1 : i32} : (tensor<75x4x94xf32>) -> tensor<75x1x94xf32>
    %11 = tosa.arithmetic_right_shift %9, %5 {round = false} : (tensor<71x21x82x34x5xi1>, tensor<71x21x82x34x5xi1>) -> tensor<71x21x82x34x5xi1>
    %12 = tosa.reduce_max %8 {axis = 0 : i32} : (tensor<150x4x94xf32>) -> tensor<1x4x94xf32>
    %13 = tosa.reduce_any %arg4 {axis = 0 : i32} : (tensor<56x91xi1>) -> tensor<1x91xi1>
    %14 = tosa.reduce_sum %12 {axis = 0 : i32} : (tensor<1x4x94xf32>) -> tensor<1x4x94xf32>
    %15 = tosa.logical_not %13 : (tensor<1x91xi1>) -> tensor<1x91xi1>
    %16 = tosa.reverse %10 {axis = 2 : i32} : (tensor<75x1x94xf32>) -> tensor<75x1x94xf32>
    %17 = tosa.clz %13 : (tensor<1x91xi1>) -> tensor<1x91xi1>
    %18 = tosa.intdiv %1, %3 : (tensor<74xi32>, tensor<74xi32>) -> tensor<74xi32>
    %19 = tosa.rsqrt %10 : (tensor<75x1x94xf32>) -> tensor<75x1x94xf32>
    return %11, %14, %15, %16, %17, %18, %19 : tensor<71x21x82x34x5xi1>, tensor<1x4x94xf32>, tensor<1x91xi1>, tensor<75x1x94xf32>, tensor<1x91xi1>, tensor<74xi32>, tensor<75x1x94xf32>
  }
}
