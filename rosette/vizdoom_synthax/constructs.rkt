(define-synthax Action
  ([(Action previous_actions)
      (append previous_actions (list (choose 0 1 2 3 4 5 6 7 8 9 10)))]))

(define-synthax (Action_Block previous_actions length)
  #:base previous_actions
  #:else (choose
          previous_actions
          ; Two blocks
          (let ([new_actions (Action_Block previous_actions (- length 1))])
            (Action new_actions))))

(define-synthax Positive_Condition
  ([(Positive_Condition previous_actions perceptions)
       (list-ref (list-ref perceptions (length previous_actions)) (choose 0 1 2 3 4 5))]))

(define-synthax Condition
  ([(Condition previous_actions perceptions)
      (choose
       (list-ref (list-ref perceptions (length previous_actions)) (choose 0 1 2 3 4 5))
       (not
        (list-ref (list-ref perceptions (length previous_actions)) (choose 0 1 2 3 4 5))))]))

(define-synthax If
  ([(If previous_actions perceptions)
           (if (Positive_Condition previous_actions perceptions)
              (Action_Block previous_actions 3)
              (Action_Block previous_actions 3))
           ]))

(define-synthax While
  ([(While previous_actions perceptions)
      (let loop ((aux1 previous_actions) (len 3) )
        (if (and (Condition aux1 perceptions) (>= len 0))
          (loop (Action_Block aux1 3) (- len 1))
          aux1))]))



