module {
  func.func @main(%arg0: tensor<25x85x45xi64>, %arg1: tensor<25x45x28xi64>, %arg2: tensor<23x14x99x6x72x75xf32>, %arg3: tensor<57x23xi1>) -> (tensor<25x85x28xi64>, tensor<23x14x99x6x72x75xf32>, tensor<25x85x28xi64>, tensor<690x1x1496880xf32>, tensor<57x23xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<25x85x45xi64>, tensor<25x45x28xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<25x85x28xi64>
    %1 = tosa.identity %0 : (tensor<25x85x28xi64>) -> tensor<25x85x28xi64>
    %2 = tosa.add %1, %1 : (tensor<25x85x28xi64>, tensor<25x85x28xi64>) -> tensor<25x85x28xi64>
    %3 = tosa.floor %arg2 : (tensor<23x14x99x6x72x75xf32>) -> tensor<23x14x99x6x72x75xf32>
    %4 = tosa.maximum %2, %1 : (tensor<25x85x28xi64>, tensor<25x85x28xi64>) -> tensor<25x85x28xi64>
    %5 = tosa.clz %4 : (tensor<25x85x28xi64>) -> tensor<25x85x28xi64>
    %6 = tosa.maximum %5, %5 : (tensor<25x85x28xi64>, tensor<25x85x28xi64>) -> tensor<25x85x28xi64>
    %7 = tosa.log %3 : (tensor<23x14x99x6x72x75xf32>) -> tensor<23x14x99x6x72x75xf32>
    %8 = tosa.logical_left_shift %2, %5 : (tensor<25x85x28xi64>, tensor<25x85x28xi64>) -> tensor<25x85x28xi64>
    %9 = tosa.logical_not %arg3 : (tensor<57x23xi1>) -> tensor<57x23xi1>
    %r_10 = tosa.const_shape {values = dense<[ 690, 1, 1496880 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %10 = tosa.reshape %3, %r_10 : (tensor<23x14x99x6x72x75xf32>, !tosa.shape<3>) -> tensor<690x1x1496880xf32>
    %11 = tosa.ceil %10 : (tensor<690x1x1496880xf32>) -> tensor<690x1x1496880xf32>
    %12 = tosa.logical_and %9, %9 : (tensor<57x23xi1>, tensor<57x23xi1>) -> tensor<57x23xi1>
    return %6, %7, %8, %11, %12 : tensor<25x85x28xi64>, tensor<23x14x99x6x72x75xf32>, tensor<25x85x28xi64>, tensor<690x1x1496880xf32>, tensor<57x23xi1>
  }
}
