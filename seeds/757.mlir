module {
  func.func @main(%arg0: tensor<78x82x14x60xi64>, %arg1: tensor<78x1x1x60xi64>, %arg2: tensor<15x19x96x36x57xf32>, %arg3: tensor<20x85xi32>, %arg4: tensor<20x85xi32>) -> (tensor<78x164x1x60xi1>, tensor<78x82x14x60xi1>, tensor<15x19x96x36x57xi1>, tensor<1x85xi1>, tensor<15x19x96x36x57xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<78x82x14x60xi64>, tensor<78x1x1x60xi64>) -> tensor<78x82x14x60xi64>
    %1 = tosa.bitwise_or %0, %0 : (tensor<78x82x14x60xi64>, tensor<78x82x14x60xi64>) -> tensor<78x82x14x60xi64>
    %2 = tosa.sub %1, %0 : (tensor<78x82x14x60xi64>, tensor<78x82x14x60xi64>) -> tensor<78x82x14x60xi64>
    %3 = tosa.concat %2, %0 {axis = 1 : i32} : (tensor<78x82x14x60xi64>, tensor<78x82x14x60xi64>) -> tensor<78x164x14x60xi64>
    %4 = tosa.reduce_product %3 {axis = 2 : i32} : (tensor<78x164x14x60xi64>) -> tensor<78x164x1x60xi64>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<78x164x1x60xi64>, tensor<78x164x1x60xi64>) -> tensor<78x164x1x60xi64>
    %6 = tosa.bitwise_not %5 : (tensor<78x164x1x60xi64>) -> tensor<78x164x1x60xi64>
    %7 = tosa.log %arg2 : (tensor<15x19x96x36x57xf32>) -> tensor<15x19x96x36x57xf32>
    %8 = tosa.greater_equal %7, %7 : (tensor<15x19x96x36x57xf32>, tensor<15x19x96x36x57xf32>) -> tensor<15x19x96x36x57xi1>
    %9 = tosa.clz %6 : (tensor<78x164x1x60xi64>) -> tensor<78x164x1x60xi64>
    %10 = tosa.greater %9, %4 : (tensor<78x164x1x60xi64>, tensor<78x164x1x60xi64>) -> tensor<78x164x1x60xi1>
    %11 = tosa.greater_equal %0, %1 : (tensor<78x82x14x60xi64>, tensor<78x82x14x60xi64>) -> tensor<78x82x14x60xi1>
    %12 = tosa.intdiv %arg3, %arg4 : (tensor<20x85xi32>, tensor<20x85xi32>) -> tensor<20x85xi32>
    %13 = tosa.greater %12, %12 : (tensor<20x85xi32>, tensor<20x85xi32>) -> tensor<20x85xi1>
    %14 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %15 = tosa.transpose %13 {perms = array<i32: 0, 1>} : (tensor<20x85xi1>) -> tensor<20x85xi1>
    %16 = tosa.add %8, %8 : (tensor<15x19x96x36x57xi1>, tensor<15x19x96x36x57xi1>) -> tensor<15x19x96x36x57xi1>
    %17 = tosa.reduce_sum %15 {axis = 0 : i32} : (tensor<20x85xi1>) -> tensor<1x85xi1>
    %18 = tosa.equal %7, %7 : (tensor<15x19x96x36x57xf32>, tensor<15x19x96x36x57xf32>) -> tensor<15x19x96x36x57xi1>
    return %10, %11, %16, %17, %18 : tensor<78x164x1x60xi1>, tensor<78x82x14x60xi1>, tensor<15x19x96x36x57xi1>, tensor<1x85xi1>, tensor<15x19x96x36x57xi1>
  }
}
