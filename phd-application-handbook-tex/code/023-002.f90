! 子程序：计算给定路径的Berry相位
  subroutine  berryphase
      !> 计算给定路径Berry相位的子程序
      !
      ! 注释：
      !
      !          目前，必须在kpoints中定义想要的k路径
      !
      ! 作者：QuanSheng Wu (wuquansheng@gmail.com)
      !
      ! 2016年3月31日
      !
      ! 版权所有 (c) 2010 QuanSheng Wu

      use para      ! 引入参数模块
      use wmpi      ! 引入MPI并行模块
      implicit none ! 显式声明所有变量类型

      ! 局部变量声明
      integer :: i, j, it, ik, Nk_seg, NK_Berry_tot  ! 循环计数器和数组维度

      !> k点在kx-ky平面中的位置 (3维坐标, 总k点数)
      real(dp), allocatable :: kpoints(:, :)

      !> 每个k点的哈密顿量
      !> 以及eigensystem_c对角化后的哈密顿量本征向量
      complex(dp), allocatable :: uk(:, :), uk_dag(:, :)  ! uk: 本征向量矩阵, uk_dag: 其厄米共轭

      !> 每个k点的本征向量 (能带×能带×k点)
      complex(dp), allocatable :: Eigenvector(:, :, :)

      ! 本征值数组和相位数组
      real   (dp), allocatable :: eigenvalue(:)     ! 哈密顿量本征值
      complex(dp), allocatable :: phase(:)          ! Berry相位累积因子
      real(dp) :: br                                ! Wannier中心投影的临时变量
      real(dp) :: k(3), b(3)                        ! k点坐标和路径段向量
      complex(dp) :: overlap, ratio                  ! 重叠积分和相位修正因子

      ! 每段k点的数量和Berry相位计算的总k点数
      Nk_seg= Nk                                    ! 每个路径段的k点数
      NK_Berry_tot= (NK_Berry-1)*Nk_seg             ! 闭合路径总k点数

      ! 分配内存
      allocate(kpoints(3, NK_Berry_tot))            ! k点坐标数组
      kpoints= 0d0                                  ! 初始化为零

      ! 分配哈密顿量和本征向量相关数组
      allocate(uk(Num_wann, Num_wann),  uk_dag(Num_wann, Num_wann))  ! 单点哈密顿量和其厄米共轭
      allocate(Eigenvector(Num_wann, Num_wann, NK_Berry_tot))       ! 所有k点的本征向量
      allocate(eigenvalue(Num_wann))                ! 本征值数组
      allocate(phase(Num_wann))                     ! Berry相位累积因子
      
      ! 初始化数组
      uk=0d0
      eigenvalue=0d0
      Eigenvector=0d0

      !> 设置Berry相位计算的k路径
      !> kpoints, k3points_Berry是分数坐标/直接坐标
      it = 0                                        ! k点索引初始化
      do ik=1, NK_Berry- 1                          ! 遍历路径段
         do i= 1, Nk_seg                           ! 在每个路径段内均匀采样
            it= it+ 1
            ! 线性插值构建路径上的k点
            kpoints(:, it)= k3points_Berry(:, ik)+ &
               (k3points_Berry(:, ik+1)- k3points_Berry(:, ik))*(i-1d0)/(Nk_seg-1d0)
         enddo ! i
      enddo ! ik

      !> 对每个k点计算Wannier中心相关的Berry相位
      do ik=1, NK_Berry_tot
         k= kpoints(:, ik)                         ! 当前k点坐标

         ! 根据计算模型选择不同的哈密顿量构造方法
         if (index(KPorTB, 'KP')/=0)then            ! 如果使用kp模型
            call ham_bulk_kp (k, uk)               ! 调用kp模型的体态哈密顿量
         else                                      ! 否则使用晶格规范
            call ham_bulk_latticegauge(k, uk)       ! 调用晶格规范的体态哈密顿量
         endif
        
         !> 对角化哈密顿量获取本征向量
         call eigensystem_c('V', 'U', Num_wann, uk, eigenvalue)

         ! 存储当前k点的本征向量
         Eigenvector(:, :, ik)= uk
      enddo

      !> 沿k路径求和得到Berry相位
      phase= 1d0                                     ! 初始化相位累积因子为1
      do ik= 1, NK_Berry_tot-1                       ! 遍历相邻k点对
         uk= Eigenvector(:, :, ik)                   ! 当前k点的本征向量
         uk_dag= conjg(transpose(uk))                ! 计算其厄米共轭 (bra向量)
         
         ! 处理闭合路径：最后一个点与起点连接
         if (ik==NK_Berry_tot-1) then
            uk= Eigenvector(:, :, 1)                ! 最后一个段连接到起点
         else
            uk= Eigenvector(:, :, ik+ 1)            ! 普通情况连接到下一个点
         endif
         
         b= kpoints(:, ik+1)- kpoints(:, ik)          ! 计算k路径段向量

         !> 计算 <u_k|u_{k+1}> 重叠积分，考虑Wannier中心的规范变换
         do i=1, Num_wann                            ! 遍历能带
            overlap= 0d0                             ! 初始化重叠积分
            
            do j=1, Num_wann                        ! 遍历能带求和
               ! 计算Wannier中心在路径段方向上的投影
               br= b(1)*Origin_cell%wannier_centers_direct(1, j)+ &
                   b(2)*Origin_cell%wannier_centers_direct(2, j)+ &
                   b(3)*Origin_cell%wannier_centers_direct(3, j)
               ! 规范变换相位因子：exp(i2π br)
               ratio= cos(2d0*pi*br)- zi*sin(2d0*pi*br)

               ! 计算矩阵元素乘积并累加
               overlap= overlap+ uk_dag(i, j)* uk(j, i)* ratio
            enddo
            
            ! 累积相位因子
            phase(i)= overlap*phase(i)
         enddo

      enddo  !< ik 完成所有路径段的积分

      ! 输出警告信息和计算结果
      if (cpuid==0)write(stdout, *) ">> WARNING: Please increase NK1 until Berry phase is converged!"
      if (cpuid==0)write(stdout, *) ">> WARNING: The starting point and the ending point should be different by a reciprocal lattice vector"
      if (cpuid==0)write(stdout, *) 'Berry phase for the loop you chose: in unit of \pi'
      ! 计算占据能带的Berry相位（模2π，以π为单位）
      if (cpuid==0) write(stdout, '(f18.6)') mod(sum(aimag(log(phase(1:NumOccupied)))/pi), 2d0)

      ! 输出详细数据到文件
      outfileindex= outfileindex+ 1
      if (cpuid==0) then
         open(unit= outfileindex, file="kpath_berry.txt")
         ! 写入文件头，包含Berry相位结果
         write(outfileindex, '("#",a11, 5a12, a, f12.6, a)')"kx", "ky", "kz", &
                              "k1", "k2", "k3", " Berry phase= ", &
                              mod(sum(aimag(log(phase(1:NumOccupied)))/pi), 2d0), ' pi'
         ! 写入所有k点的坐标
         do ik=1, NK_Berry_tot
            ! 将分数坐标转换为笛卡尔坐标
            b= kpoints(1, ik)*Origin_cell%Kua+ kpoints(2, ik)*Origin_cell%Kub+ kpoints(3, ik)*Origin_cell%Kuc
            write(outfileindex, '(6f12.6)')b, kpoints(:, ik)
         enddo
      endif
      
      ! 释放动态分配的内存
      deallocate(kpoints)
      deallocate(uk,uk_dag)
      deallocate(Eigenvector,eigenvalue)
      deallocate(phase)
 
      return
   end subroutine berryphase

  ! 子程序：使用原子规范计算Berry相位（改进版本）
  subroutine  berryphase_atomic
      !> 使用原子规范计算给定路径Berry相位的子程序
      ! [头部注释与上一子程序相同，此处省略]

      use para
      use wmpi
      implicit none

      ! 局部变量（与上一子程序类似，增加mat1, mat2）
      integer :: i, j, it, ik, Nk_seg, NK_Berry_tot

      !> k点在kx-ky平面中的位置
      real(dp), allocatable :: kpoints(:, :)

      !> 哈密顿量和本征向量
      complex(dp), allocatable :: uk(:, :), uk_dag(:, :)
      complex(dp), allocatable :: Eigenvector(:, :, :)

      ! 本征值和相关矩阵
      real   (dp), allocatable :: eigenvalue(:)
      complex(dp), allocatable :: phase(:), mat1(:, :), mat2(:, :)  ! 新增工作矩阵
      real(dp) :: br
      real(dp) :: k(3), b(3)
      complex(dp) :: overlap, ratio

      ! 路径参数设置（与上一子程序相同）
      Nk_seg= Nk
      NK_Berry_tot= (NK_Berry-1)*Nk_seg 

      ! 分配内存（额外分配mat1, mat2用于中间计算）
      allocate(kpoints(3, NK_Berry_tot))
      kpoints= 0d0

      allocate(mat1(Num_wann, Num_wann), mat2(Num_wann, Num_wann))  ! 工作矩阵
      allocate(uk(Num_wann, Num_wann),  uk_dag(Num_wann, Num_wann))
      allocate(Eigenvector(Num_wann, Num_wann, NK_Berry_tot), eigenvalue(Num_wann))
      allocate(phase(Num_wann))
      
      ! 初始化
      uk=0d0
      eigenvalue=0d0
      Eigenvector=0d0

      !> 设置k路径（与上一子程序完全相同）
      it = 0
      do ik=1, NK_Berry- 1
         do i= 1, Nk_seg
            it= it+ 1
            kpoints(:, it)= k3points_Berry(:, ik)+ &
               (k3points_Berry(:, ik+1)- k3points_Berry(:, ik))*(i-1d0)/(Nk_seg-1d0)
         enddo ! i
      enddo ! ik

      !> 计算每个k点的本征向量（关键区别：使用原子规范）
      do ik=1, NK_Berry_tot
         k= kpoints(:, ik)

         ! 主要区别：使用原子规范而非晶格规范或kp模型
         if (index(KPorTB, 'KP')/=0)then
            call ham_bulk_kp (k, uk)               ! kp模型（如果适用）
         else
            call ham_bulk_atomicgauge(k, uk)       ! 原子规范哈密顿量
         endif
        
         !> 对角化（与上一子程序相同）
         call eigensystem_c('V', 'U', Num_wann, uk, eigenvalue)

         Eigenvector(:, :, ik)= uk
      enddo

      !> 计算Berry相位（主要逻辑改进）
      phase= 1d0
      do ik= 1, NK_Berry_tot-1
         uk= Eigenvector(:, :, ik)
         uk_dag= conjg(transpose(uk))
         
         ! 改进的路径闭合处理
         if (ik==NK_Berry_tot-1) then
            ! 显式计算闭合向量（从终点到起点）
            b=-kpoints(:, 1)+ kpoints(:, NK_Berry_tot)
            uk= Eigenvector(:, :, 1)
         else
            uk= Eigenvector(:, :, ik+ 1)
         endif

         !> 改进的重叠积分计算
         do i=1, Num_wann
            overlap= 0d0
            do j=1, Num_wann
               if (ik==NK_Berry_tot-1) then
                  ! 只在闭合段应用Wannier中心规范变换
                  br= b(1)*Origin_cell%wannier_centers_direct(1, j)+ &
                      b(2)*Origin_cell%wannier_centers_direct(2, j)+ &
                      b(3)*Origin_cell%wannier_centers_direct(3, j)
                  ratio= cos(2d0*pi*br)- zi*sin(2d0*pi*br)
               else
                  ! 非闭合段不使用规范变换（ratio=1）
                  ratio=1d0
               endif

               overlap= overlap+ uk_dag(i, j)* uk(j, i)* ratio
            enddo
            phase(i)= overlap*phase(i)
         enddo

      enddo  !< ik

      ! 输出部分（与上一子程序完全相同）
      if (cpuid==0)write(stdout, *) ">> WARNING: Please increase NK1 until Berry phase is converged!"
      if (cpuid==0)write(stdout, *) ">> WARNING: The starting point and the ending point should be different by a reciprocal lattice vector"
      if (cpuid==0)write(stdout, *) 'Berry phase for the loop you chose: in unit of \pi'
      if (cpuid==0) write(stdout, '(f18.6)') mod(sum(aimag(log(phase(1:NumOccupied)))/pi), 2d0)

      outfileindex= outfileindex+ 1
      if (cpuid==0) then
         open(unit= outfileindex, file="kpath_berry.txt")
         write(outfileindex, '("#",a11, 5a12, a, f12.6, a)')"kx", "ky", "kz", &
                              "k1", "k2", "k3", " Berry phase= ", &
                              mod(sum(aimag(log(phase(1:NumOccupied)))/pi), 2d0), ' pi'
         do ik=1, NK_Berry_tot
            b= kpoints(1, ik)*Origin_cell%Kua+ kpoints(2, ik)*Origin_cell%Kub+ kpoints(3, ik)*Origin_cell%Kuc
            write(outfileindex, '(6f12.6)')b, kpoints(:, ik)
         enddo
      endif
      
      ! 释放内存（包括新增的mat1, mat2）
      deallocate(kpoints)
      deallocate(mat1,mat2,uk,uk_dag)
      deallocate(Eigenvector,eigenvalue)
      deallocate(phase)
 
      return
   end subroutine berryphase_atomic
