module {
  func.func @main(%arg0: tensor<6x86x61x24x97x22xi8>, %arg1: tensor<6x86x1x1x97x22xi8>, %arg2: tensor<27x62x23x64x48xf32>, %arg3: tensor<21xi1>) -> (tensor<6x86x61x24x97x22xi8>, tensor<1xi1>, tensor<27x62x23x64x48xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<6x86x61x24x97x22xi8>, tensor<6x86x1x1x97x22xi8>) -> tensor<6x86x61x24x97x22xi8>
    %1 = tosa.tanh %arg2 : (tensor<27x62x23x64x48xf32>) -> tensor<27x62x23x64x48xf32>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<6x86x61x24x97x22xi8>, tensor<6x86x61x24x97x22xi8>) -> tensor<6x86x61x24x97x22xi8>
    %3 = tosa.reduce_product %arg3 {axis = 0 : i32} : (tensor<21xi1>) -> tensor<1xi1>
    %4 = tosa.equal %1, %1 : (tensor<27x62x23x64x48xf32>, tensor<27x62x23x64x48xf32>) -> tensor<27x62x23x64x48xi1>
    return %2, %3, %4 : tensor<6x86x61x24x97x22xi8>, tensor<1xi1>, tensor<27x62x23x64x48xi1>
  }
}
