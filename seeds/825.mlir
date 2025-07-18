module {
  func.func @main(%arg0: tensor<20x54xi64>, %arg1: tensor<1x1xi64>, %arg2: tensor<8x48x47x42xi1>, %arg3: tensor<8x48x47x42xi1>, %arg4: tensor<87xf32>) -> (tensor<20xi32>, tensor<174xf32>, tensor<8x48x47x42xi1>, tensor<87xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<20x54xi64>, tensor<1x1xi64>) -> tensor<20x54xi64>
    %1 = tosa.argmax %0 {axis = 1 : i32} : (tensor<20x54xi64>) -> tensor<20xi32>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<8x48x47x42xi1>, tensor<8x48x47x42xi1>) -> tensor<8x48x47x42xi1>
    %3 = tosa.bitwise_and %2, %2 : (tensor<8x48x47x42xi1>, tensor<8x48x47x42xi1>) -> tensor<8x48x47x42xi1>
    %4 = tosa.logical_not %3 : (tensor<8x48x47x42xi1>) -> tensor<8x48x47x42xi1>
    %5 = tosa.identity %3 : (tensor<8x48x47x42xi1>) -> tensor<8x48x47x42xi1>
    %6 = tosa.clz %4 : (tensor<8x48x47x42xi1>) -> tensor<8x48x47x42xi1>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %7 = tosa.negate %5, %in_zp_7, %out_zp_7 : (tensor<8x48x47x42xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<8x48x47x42xi1>
    %8 = tosa.reciprocal %arg4 : (tensor<87xf32>) -> tensor<87xf32>
    %9 = tosa.concat %8, %8 {axis = 0 : i32} : (tensor<87xf32>, tensor<87xf32>) -> tensor<174xf32>
    %10 = tosa.clz %6 : (tensor<8x48x47x42xi1>) -> tensor<8x48x47x42xi1>
    %11 = tosa.bitwise_and %10, %7 : (tensor<8x48x47x42xi1>, tensor<8x48x47x42xi1>) -> tensor<8x48x47x42xi1>
    %12 = tosa.abs %11 : (tensor<8x48x47x42xi1>) -> tensor<8x48x47x42xi1>
    %13 = tosa.pow %8, %8 : (tensor<87xf32>, tensor<87xf32>) -> tensor<87xf32>
    return %1, %9, %12, %13 : tensor<20xi32>, tensor<174xf32>, tensor<8x48x47x42xi1>, tensor<87xf32>
  }
}
