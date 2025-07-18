module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<78x67x29x26xi64>, %arg3: tensor<17x27x25x66x90x82xf32>) -> (tensor<i64>, tensor<17x27x25x66x90x82xf32>, tensor<804x1x1xi32>, tensor<17x27x25x66x90x82xf32>, tensor<804x169x1xi32>, tensor<17x27x25x66x90x82xf32>, tensor<2412x338x3xi1>, tensor<17x27x25x66x90x82xf32>, tensor<804x169x1xi32>, tensor<1608x1x2xi32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %1 = tosa.argmax %arg2 {axis = 2 : i32} : (tensor<78x67x29x26xi64>) -> tensor<78x67x26xi32>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %2 = tosa.negate %0, %in_zp_2, %out_zp_2 : (tensor<i64>, tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
    %3 = tosa.add %2, %2 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = tosa.logical_left_shift %1, %1 : (tensor<78x67x26xi32>, tensor<78x67x26xi32>) -> tensor<78x67x26xi32>
    %r_5 = tosa.const_shape {values = dense<[ 804, 169, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.reshape %4, %r_5 : (tensor<78x67x26xi32>, !tosa.shape<3>) -> tensor<804x169x1xi32>
    %6 = tosa.tanh %arg3 : (tensor<17x27x25x66x90x82xf32>) -> tensor<17x27x25x66x90x82xf32>
    %7 = tosa.log %6 : (tensor<17x27x25x66x90x82xf32>) -> tensor<17x27x25x66x90x82xf32>
    %8 = tosa.rsqrt %7 : (tensor<17x27x25x66x90x82xf32>) -> tensor<17x27x25x66x90x82xf32>
    %9 = tosa.reduce_max %5 {axis = 1 : i32} : (tensor<804x169x1xi32>) -> tensor<804x1x1xi32>
    %10 = tosa.bitwise_or %5, %5 : (tensor<804x169x1xi32>, tensor<804x169x1xi32>) -> tensor<804x169x1xi32>
    %11 = tosa.exp %6 : (tensor<17x27x25x66x90x82xf32>) -> tensor<17x27x25x66x90x82xf32>
    %12 = tosa.logical_left_shift %10, %5 : (tensor<804x169x1xi32>, tensor<804x169x1xi32>) -> tensor<804x169x1xi32>
    %13 = tosa.logical_right_shift %12, %10 : (tensor<804x169x1xi32>, tensor<804x169x1xi32>) -> tensor<804x169x1xi32>
    %14 = tosa.log %6 : (tensor<17x27x25x66x90x82xf32>) -> tensor<17x27x25x66x90x82xf32>
    %15 = tosa.exp %14 : (tensor<17x27x25x66x90x82xf32>) -> tensor<17x27x25x66x90x82xf32>
    %16 = tosa.exp %15 : (tensor<17x27x25x66x90x82xf32>) -> tensor<17x27x25x66x90x82xf32>
    %17 = tosa.reduce_min %10 {axis = 2 : i32} : (tensor<804x169x1xi32>) -> tensor<804x169x1xi32>
    %t_18 = tosa.const_shape {values = dense<[ 3, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %18 = tosa.tile %17, %t_18 : (tensor<804x169x1xi32>, !tosa.shape<3>) -> tensor<2412x338x3xi32>
    %19 = tosa.greater_equal %18, %18 : (tensor<2412x338x3xi32>, tensor<2412x338x3xi32>) -> tensor<2412x338x3xi1>
    %20 = tosa.reverse %10 {axis = 0 : i32} : (tensor<804x169x1xi32>) -> tensor<804x169x1xi32>
    %21 = tosa.log %6 : (tensor<17x27x25x66x90x82xf32>) -> tensor<17x27x25x66x90x82xf32>
    %22 = tosa.clz %20 : (tensor<804x169x1xi32>) -> tensor<804x169x1xi32>
    %t_23 = tosa.const_shape {values = dense<[ 2, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %23 = tosa.tile %20, %t_23 : (tensor<804x169x1xi32>, !tosa.shape<3>) -> tensor<1608x507x2xi32>
    %24 = tosa.reduce_product %23 {axis = 1 : i32} : (tensor<1608x507x2xi32>) -> tensor<1608x1x2xi32>
    return %3, %8, %9, %11, %13, %16, %19, %21, %22, %24 : tensor<i64>, tensor<17x27x25x66x90x82xf32>, tensor<804x1x1xi32>, tensor<17x27x25x66x90x82xf32>, tensor<804x169x1xi32>, tensor<17x27x25x66x90x82xf32>, tensor<2412x338x3xi1>, tensor<17x27x25x66x90x82xf32>, tensor<804x169x1xi32>, tensor<1608x1x2xi32>
  }
}
